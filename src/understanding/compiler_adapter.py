from __future__ import annotations
import re
import yaml
from pathlib import Path
from src import config
from src.data_loader import normalize_vi
from src.understanding.query_plan import QueryPlan
from src.tasco_query.service import get_service
from src.tasco_query.contracts import QueryUnderstandRequest, UserLocation

# Load gazetteer mapping
_GAZETTEER_FILE = Path(__file__).resolve().parents[2] / "lexicon" / "gazetteer.yaml"
_GAZ_DATA = yaml.safe_load(_GAZETTEER_FILE.read_text(encoding="utf-8")) or {}

def _match_landmark(poi_name: str | None) -> str | None:
    if not poi_name:
        return None
    norm = normalize_vi(poi_name)
    # Match against names in gazetteer
    for key, entry in _GAZ_DATA.items():
        for name in entry.get("names", []):
            if normalize_vi(name) in norm or norm in normalize_vi(name):
                return key
    return None

def compile_query_to_plan(query: str, user_coord: tuple[float, float] | None = None, attr_index=None, joint_index=None, column_anchor=None) -> QueryPlan:
    # 1. Start with baseline plan to prevent regressions
    from src.understanding.rules import extract_plan as extract_plan_baseline
    plan = extract_plan_baseline(query, attr_index=attr_index, joint_index=joint_index, column_anchor=column_anchor)
    
    # 2. Extract advanced properties from tasco_query service
    service = get_service()
    location_payload = None
    if user_coord is not None:
        location_payload = UserLocation(lat=user_coord[0], lon=user_coord[1])
        
    req = QueryUnderstandRequest(query=query, location=location_payload)
    try:
        result = service.understand(req)
        response = result.response
        entities = response.entities
        
        # Merge categories with appropriate mappings
        category = entities.get("category")
        if category:
            category_mapping = {
                "cây xăng": "Trạm xăng",
                "rạp chiếu phim": "Rạp phim",
                "rạp phim": "Rạp phim",
                "quán cà phê": "Quán cà phê",
                "quán cafe": "Quán cà phê",
                "nhà hàng": "Nhà hàng",
                "quán ăn": "Nhà hàng",
                "khách sạn": "Khách sạn",
                "bệnh viện": "Bệnh viện",
                "nhà thuốc": "Nhà thuốc",
                "trung tâm thương mại": "Trung tâm thương mại",
                "công viên": "Công viên",
                "atm": "ATM",
                "trạm sạc điện": "Trạm sạc điện"
            }
            mapped_cat = category_mapping.get(normalize_vi(category))
            if mapped_cat:
                plan.categories.add(mapped_cat)
                
        if entities.get("cuisine") or entities.get("dish") or entities.get("destination_category") == "restaurant":
            plan.categories.add("Nhà hàng")
            
        # Merge city & district
        if not plan.city and entities.get("city"):
            city = entities["city"]
            if "hồ chí minh" in city.lower() or "ho chi minh" in city.lower():
                plan.city = "TP.HCM"
            else:
                plan.city = city
                
        if not plan.district and entities.get("district"):
            plan.district = entities["district"]
            
        # Merge landmark
        landmark_candidate = (
            entities.get("reference_poi") 
            or entities.get("poi_name") 
            or entities.get("destination_poi") 
            or entities.get("reference_area")
        )
        landmark_key = _match_landmark(landmark_candidate)
        if landmark_key:
            plan.landmark = plan.landmark or landmark_key
            if not plan.resolved_coord:
                plan.resolved_coord = (_GAZ_DATA[landmark_key]["lat"], _GAZ_DATA[landmark_key]["lon"])
            plan.city = plan.city or _GAZ_DATA[landmark_key]["city"]
            
        if not plan.resolved_coord and entities.get("latitude") is not None and entities.get("longitude") is not None:
            plan.resolved_coord = (entities["latitude"], entities["longitude"])
            
        # Merge attributes & amenities
        from src.understanding.rules import concept_tokens
        c_tokens = concept_tokens()
        
        attributes_and_amenities = []
        if entities.get("attributes"):
            attributes_and_amenities.extend(entities["attributes"])
        if entities.get("amenities"):
            attributes_and_amenities.extend(entities["amenities"])
            
        if entities.get("car_accessible") or entities.get("transport_mode") == "car":
            plan.attr_concepts.add("bai_do_xe")
        if entities.get("privacy_preference") == "high":
            plan.attr_concepts.add("yen_tinh")
            plan.attr_concepts.add("phong_hop")
            
        for attr in attributes_and_amenities:
            norm_attr = normalize_vi(attr)
            matched_concept = None
            for cid, tokens in c_tokens.items():
                if norm_attr in tokens or any(norm_attr in t or t in norm_attr for t in tokens):
                    matched_concept = cid
                    break
            if matched_concept:
                plan.attr_concepts.add(matched_concept)
                
        if entities.get("open_late") or entities.get("open_until") or entities.get("open_after"):
            plan.attr_concepts.add("mo_khuya")
        if entities.get("open_24h"):
            plan.attr_concepts.add("hai_bon_bay")
        if entities.get("price_max") or entities.get("sentiment") == "rẻ":
            plan.attr_concepts.add("gia_re")
            
        # Check polarity
        norm_query_low = plan.norm_query.lower()
        from src.understanding.rules import _NEG_QUIET
        if _NEG_QUIET.search(norm_query_low):
            plan.attr_concepts.add("yen_tinh")
            plan.neg_concepts.add("dong_khach")
            
        plan.attr_concepts -= plan.neg_concepts
        
        from src.understanding.rules import _POP
        plan.want_pop = plan.want_pop or bool(_POP.search(plan.norm_query))
        
    except Exception:
        pass
        
    return plan

def get_plan_and_normalized_query(raw_query: str, user_coord: tuple[float, float] | None = None, attr_index=None, joint_index=None, column_anchor=None) -> tuple[QueryPlan, str]:
    from src.ranking.reranker import preprocess_query
    norm = preprocess_query(raw_query)
    if config.USE_ADVANCED_COMPILER:
        plan = compile_query_to_plan(raw_query, user_coord, attr_index=attr_index, joint_index=joint_index, column_anchor=column_anchor)
        plan.norm_query = norm
        return plan, norm
    else:
        from src.understanding.rules import extract_plan
        plan = extract_plan(norm, attr_index=attr_index, joint_index=joint_index, column_anchor=column_anchor)
        if user_coord is not None and plan.resolved_coord is None:
            plan.resolved_coord = user_coord
        return plan, norm

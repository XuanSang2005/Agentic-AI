// Tasco Maps — SDK adapter mẫu cho Flutter app (AABW 2026 Track 2).
//
// Map PlaceResult (docs/tasco_api.pdf, Common DTOs) → SearchSuggestion của app:
//   id                  → SearchSuggestion.id
//   label | name        → SearchSuggestion.label
//   category | type     → SearchSuggestion.meta
//   address             → SearchSuggestion.description
//   coordinates.lat/lon → SearchSuggestion.coordinates
//
// Theo "Compatibility requirements" của PDF: constructor cấu hình được baseUrl +
// bearerToken / apiKey / headerProvider — KHÔNG hardcode credentials; không phụ
// thuộc UI layer. Đây là REST-example, dùng package:http.

import 'dart:convert';
import 'package:http/http.dart' as http;

/// Model tối giản khớp SearchSuggestion của app (packages/tasco_maps_core).
class SearchSuggestion {
  final String id;
  final String label;
  final String meta;
  final String description;
  final ({double lat, double lon}) coordinates;

  const SearchSuggestion({
    required this.id,
    required this.label,
    required this.meta,
    required this.description,
    required this.coordinates,
  });
}

class TascoSearchClient {
  final String baseUrl;                       // vd: http://localhost:8000
  final String? bearerToken;                  // Authorization: Bearer <token>
  final String? apiKey;                       // X-API-Key: <key>
  final Map<String, String> Function()? headerProvider; // auth-ready network layer

  const TascoSearchClient({
    required this.baseUrl,
    this.bearerToken,
    this.apiKey,
    this.headerProvider,
  });

  Map<String, String> _headers() => {
        if (headerProvider != null) ...headerProvider!(),
        if (bearerToken != null) 'Authorization': 'Bearer $bearerToken',
        if (apiKey != null) 'X-API-Key': apiKey!,
        'X-Locale': 'vi-VN',
      };

  /// GET /v1/search — trả list SearchSuggestion đã map từ PlaceResult.
  Future<List<SearchSuggestion>> search(
    String query, {
    double? lat,
    double? lon,
    int limit = 10,
    String lang = 'vi',
  }) async {
    final uri = Uri.parse('$baseUrl/v1/search').replace(queryParameters: {
      'q': query, // Uri tự percent-encode — giữ nguyên dấu tiếng Việt
      if (lat != null) 'lat': '$lat',
      if (lon != null) 'lon': '$lon',
      'limit': '$limit',
      'lang': lang,
    });
    final res = await http.get(uri, headers: _headers());
    if (res.statusCode != 200) {
      final err = jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
      throw Exception('search failed: ${err['error']?['code']} '
          '(requestId: ${err['requestId']})');
    }
    final body = jsonDecode(utf8.decode(res.bodyBytes)) as Map<String, dynamic>;
    return (body['results'] as List).cast<Map<String, dynamic>>().map((r) {
      final coords = r['coordinates'] as Map<String, dynamic>;
      return SearchSuggestion(
        id: r['id'] as String,
        label: (r['label'] ?? r['name']) as String,
        meta: (r['category'] ?? r['type']) as String,
        description: (r['address'] ?? '') as String,
        coordinates: (
          lat: (coords['lat'] as num).toDouble(),
          lon: (coords['lon'] as num).toDouble(),
        ),
      );
    }).toList();
  }
}

/// Ví dụ dùng:
Future<void> main() async {
  final client = TascoSearchClient(
    baseUrl: 'http://localhost:8000',
    bearerToken: const String.fromEnvironment('TASCO_TOKEN'), // không hardcode
  );
  final results = await client.search(
    'quán cà phê yên tĩnh để làm việc',
    lat: 21.0287,
    lon: 105.8524,
    limit: 5,
  );
  for (final s in results) {
    print('${s.id} | ${s.label} | ${s.meta} | ${s.description}');
  }
}

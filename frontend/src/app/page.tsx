"use client";

import { useEffect, useState } from "react";
import { SearchExperience } from "@/components/search-experience";

export default function Home() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return <div className="wrap" />;
  }

  return (
    <>
      <div className="orb a" />
      <div className="orb b" />
      <div className="wrap">
        <SearchExperience />
      </div>
    </>
  );
}

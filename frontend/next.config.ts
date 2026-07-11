import type { NextConfig } from "next";
import os from "os";

const isProd = process.env.NODE_ENV === "production";

const getLocalIPs = () => {
  const interfaces = os.networkInterfaces();
  const ips: string[] = [];
  for (const name of Object.keys(interfaces)) {
    for (const net of interfaces[name] || []) {
      if (net.family === "IPv4" && !net.internal) {
        ips.push(net.address);
      }
    }
  }
  return ips;
};

const nextConfig: NextConfig = {
  reactCompiler: true,
  allowedDevOrigins: getLocalIPs(),
  ...(isProd
    ? { output: "export" }
    : {
        async rewrites() {
          const API_ORIGIN = process.env.API_ORIGIN ?? "http://localhost:8000";
          return [{ source: "/v1/:path*", destination: `${API_ORIGIN}/v1/:path*` }];
        },
      }),
};

export default nextConfig;

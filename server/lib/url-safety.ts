import dns from "dns/promises";
import type { LookupAddress } from "dns";
import net from "net";

export type RemoteUrlValidationResult =
  | { ok: true; baseUrl: string }
  | { ok: false; reason: string };

function isPrivateIpv4(ip: string): boolean {
  const parts = ip.split(".").map((p) => Number(p));
  if (parts.length !== 4 || parts.some((p) => !Number.isFinite(p) || p < 0 || p > 255)) {
    return true;
  }

  const [a, b] = parts;

  if (a === 10) return true;
  if (a === 127) return true;
  if (a === 0) return true;
  if (a === 169 && b === 254) return true;
  if (a === 172 && b >= 16 && b <= 31) return true;
  if (a === 192 && b === 168) return true;

  return false;
}

function isPrivateIpv6(ip: string): boolean {
  const normalized = ip.toLowerCase();
  if (normalized === "::1") return true;
  if (normalized.startsWith("fe80:")) return true;
  if (normalized.startsWith("fc") || normalized.startsWith("fd")) return true;

  return false;
}

function isPrivateIp(ip: string): boolean {
  const ipVersion = net.isIP(ip);
  if (ipVersion === 4) return isPrivateIpv4(ip);
  if (ipVersion === 6) return isPrivateIpv6(ip);
  return true;
}

export async function validateRemoteBaseUrl(
  rawUrl: string,
  options?: { allowPrivateNetworks?: boolean; allowHttpInProduction?: boolean }
): Promise<RemoteUrlValidationResult> {
  const allowPrivateNetworks = options?.allowPrivateNetworks === true;
  const allowHttpInProduction = options?.allowHttpInProduction === true;

  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    return { ok: false, reason: "Invalid URL" };
  }

  if (parsed.username || parsed.password) {
    return { ok: false, reason: "Credentials in URL are not allowed" };
  }

  if (parsed.protocol !== "https:" && parsed.protocol !== "http:") {
    return { ok: false, reason: "Only http/https URLs are allowed" };
  }

  const isDev = process.env.NODE_ENV === "development";
  if (!isDev && parsed.protocol === "http:" && !allowHttpInProduction) {
    return { ok: false, reason: "Insecure http URL is not allowed" };
  }

  if (!parsed.hostname) {
    return { ok: false, reason: "URL hostname is required" };
  }

  if (parsed.hostname === "localhost") {
    return allowPrivateNetworks
      ? { ok: true, baseUrl: parsed.origin }
      : { ok: false, reason: "localhost is not allowed" };
  }

  if (parsed.hostname === "0.0.0.0") {
    return { ok: false, reason: "0.0.0.0 is not allowed" };
  }

  if (parsed.hostname === "169.254.169.254") {
    return { ok: false, reason: "Link-local metadata IP is not allowed" };
  }

  const ipLiteralVersion = net.isIP(parsed.hostname);
  if (ipLiteralVersion) {
    if (!allowPrivateNetworks && isPrivateIp(parsed.hostname)) {
      return { ok: false, reason: "Private network IPs are not allowed" };
    }

    return { ok: true, baseUrl: parsed.origin };
  }

  let lookups: LookupAddress[];
  try {
    lookups = await dns.lookup(parsed.hostname, { all: true });
  } catch {
    return { ok: false, reason: "Hostname could not be resolved" };
  }

  if (!allowPrivateNetworks) {
    for (const entry of lookups) {
      if (isPrivateIp(entry.address)) {
        return { ok: false, reason: "Hostname resolves to a private network address" };
      }
    }
  }

  return { ok: true, baseUrl: parsed.origin };
}

import { chromium } from "playwright";
import { mkdirSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OUT_DIR = join(__dirname, "..", "results", "screenshots");
mkdirSync(OUT_DIR, { recursive: true });

const APP_URL = "http://localhost:5173/";

async function main() {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1600, height: 1000 } });
  page.on("console", (msg) => console.log("[console]", msg.type(), msg.text()));
  page.on("pageerror", (err) => console.log("[pageerror]", err.message));
  page.on("requestfailed", (req) => console.log("[requestfailed]", req.url(), req.failure()?.errorText));

  await page.goto(APP_URL, { waitUntil: "networkidle" });

  // wait for terminal list to populate
  await page.waitForSelector(".terminal-item", { timeout: 15000 });
  // wait for map tiles / markers to render
  await page.waitForSelector(".risk-marker", { timeout: 15000 });
  await page.waitForTimeout(2500); // let map tiles finish painting

  await page.screenshot({ path: join(OUT_DIR, "ranked_atm_heatmap.png") });
  console.log("Saved ranked_atm_heatmap.png");

  // click the top-ranked terminal to open the officer alert view
  await page.click(".terminal-item:first-child");
  await page.waitForSelector(".alert-panel", { timeout: 10000 });
  await page.waitForTimeout(500);

  await page.screenshot({ path: join(OUT_DIR, "officer_alert_view.png") });
  console.log("Saved officer_alert_view.png");

  await browser.close();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

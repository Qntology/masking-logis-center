import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const packageJsonPath = path.join(__dirname, "..", "package.json");
const extensionJsonPath = path.join(__dirname, "..", "gemini-extension.json");

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function writeJson(filePath, value) {
  fs.writeFileSync(filePath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

function isValidVersion(value) {
  return /^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$/.test(value);
}

const requestedVersion = process.argv[2]?.trim();
if (requestedVersion && !isValidVersion(requestedVersion)) {
  console.error(
    `Invalid version '${requestedVersion}'. Expected semver like 0.7.2 or 0.7.2-rc.1.`,
  );
  process.exit(1);
}

const packageJson = readJson(packageJsonPath);
const extensionJson = readJson(extensionJsonPath);

const targetVersion = requestedVersion || packageJson.version;
if (!isValidVersion(targetVersion)) {
  console.error(
    `Invalid source version '${targetVersion}' in package.json. Expected semver like 0.7.2.`,
  );
  process.exit(1);
}

packageJson.version = targetVersion;
extensionJson.version = targetVersion;

writeJson(packageJsonPath, packageJson);
writeJson(extensionJsonPath, extensionJson);

console.log(`Synchronized package + extension version to ${targetVersion}`);

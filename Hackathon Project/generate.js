const fs = require("fs");

const html_string = fs.readFileSync("./src/index.html", { encoding: "utf-8" });
const js_string = fs.readFileSync("./src/index.js", { encoding: "utf-8" });

let app = js_string.replace('"HTML_PLACEHOLDER"', "`" + html_string + "`");

fs.writeFileSync("build/app.js", app);

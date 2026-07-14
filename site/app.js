const set = (id, value) => { document.getElementById(id).textContent = value; };
const percent = (value) => `${Number(value).toFixed(1)}%`;
const ms = (value) => `${Number(value).toFixed(2)} ms`;
const mb = (value) => `${Number(value).toFixed(2)} MB`;
const rate = (value) => `${Number(value).toFixed(1)}/s`;

fetch("benchmark.json", { cache: "no-store" })
  .then((response) => {
    if (!response.ok) throw new Error("Benchmark not published yet");
    return response.json();
  })
  .then((data) => {
    if (!data.improvement) throw new Error("Benchmark is pending");
    set("status", `${data.verdict.passed ? "Quality gate passed" : "Quality gate failed"} on ${data.optimized.runtime.architecture}`);
    set("size", percent(data.improvement.model_size_reduction_percent));
    set("latency", percent(data.improvement.latency_reduction_percent));
    set("throughput", percent(data.improvement.throughput_gain_percent));
    set("cosine", Number(data.validation.cosine_similarity).toFixed(5));
    set("base-size", mb(data.baseline.model_mb));
    set("opt-size", mb(data.optimized.model_mb));
    set("base-latency", ms(data.baseline.latency.mean_ms));
    set("opt-latency", ms(data.optimized.latency.mean_ms));
    set("base-p95", ms(data.baseline.latency.p95_ms));
    set("opt-p95", ms(data.optimized.latency.p95_ms));
    set("base-throughput", rate(data.baseline.latency.throughput_items_per_second));
    set("opt-throughput", rate(data.optimized.latency.throughput_items_per_second));
    set("generated", `Generated ${new Date(data.generated_at).toLocaleString()}`);
  })
  .catch((error) => set("status", error.message));


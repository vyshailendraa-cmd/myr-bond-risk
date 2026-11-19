import { useEffect, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Cell, ComposedChart, Legend, Line, ReferenceLine,
  ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis,
} from "recharts";

const API = "/api";
const NAVY = "#1f3a5f";
const ORANGE = "#e67e22";
const GREY = "#7f8c8d";

type Row = Record<string, any>;

async function get<T = any>(path: string): Promise<T> {
  const r = await fetch(API + path);
  if (!r.ok) throw new Error(`${path}: ${r.status}`);
  return r.json();
}

const rm = (v: number) => `${v < 0 ? "-" : "+"}RM${Math.abs(v).toLocaleString("en-MY", { maximumFractionDigits: 0 })}`;
const rmPlain = (v: number) => `RM${v.toLocaleString("en-MY", { maximumFractionDigits: 0 })}`;

function Card({ title, children, note }: { title: string; children: React.ReactNode; note?: string }) {
  return (
    <section className="card">
      <h2>{title}</h2>
      {children}
      {note && <p className="note">{note}</p>}
    </section>
  );
}

function Kpi({ label, value }: { label: string; value: string }) {
  return (
    <div className="kpi">
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
    </div>
  );
}

export default function App() {
  const [err, setErr] = useState<string | null>(null);
  const [curve, setCurve] = useState<any>(null);
  const [port, setPort] = useState<any>(null);
  const [shapes, setShapes] = useState<Row[]>([]);
  const [krd, setKrd] = useState<Row[]>([]);
  const [oprGrid, setOprGrid] = useState<Row[]>([]);
  const [backtest, setBacktest] = useState<any>(null);
  const [opr, setOpr] = useState(25);
  const [oprPnl, setOprPnl] = useState<Row[]>([]);
  const [mpsText, setMpsText] = useState("");
  const [mps, setMps] = useState<any>(null);
  const [mpsBusy, setMpsBusy] = useState(false);

  useEffect(() => {
    Promise.all([
      get("/portfolio").then(setPort),
      get<Row[]>("/stress/shapes").then(setShapes),
      get<Row[]>("/stress/key-rates").then(setKrd),
      get<Row[]>("/stress/opr").then((rows) => {
        const by: Record<number, Row> = {};
        rows.forEach((r) => {
          by[r.shock_bp] = by[r.shock_bp] || { shock: `${r.shock_bp > 0 ? "+" : ""}${r.shock_bp}bp` };
          by[r.shock_bp][r.scenario] = r["pnl_%"];
        });
        setOprGrid(Object.keys(by).map(Number).sort((a, b) => a - b).map((k) => by[k]));
      }),
      get("/forecast/backtest").then(setBacktest).catch(() => setBacktest(null)),
    ]).catch((e) => setErr(String(e)));
  }, []);

  useEffect(() => {
    get(`/curve?opr_shock_bp=${opr}`).then(setCurve).catch((e) => setErr(String(e)));
    get(`/stress/opr-shock?bp=${opr}`).then((d) => setOprPnl(d.results)).catch(() => {});
  }, [opr]);

  async function runMps() {
    setMpsBusy(true);
    setMps(null);
    try {
      const r = await fetch(API + "/mps/summary", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: mpsText }),
      });
      const d = await r.json();
      setMps(r.ok ? d : { error: d.detail || "Request failed" });
    } finally {
      setMpsBusy(false);
    }
  }

  if (err) return <div className="page"><p className="error">Could not reach the API. Is it running on port 8000? ({err})</p></div>;
  if (!curve || !port) return <div className="page"><p>Loading…</p></div>;

  const s = port.summary;
  const chartData = curve.fitted.map((p: Row) => ({ t: p.t, fit: p.fit, shocked: p.shocked }));
  const marketPts = curve.market.map((m: Row) => ({ t: m.maturity_years, market: m.yield_pct }));
  const pass = oprPnl.find((r) => r.scenario === "OPR pass-through");
  const par = oprPnl.find((r) => r.scenario === "parallel");
  const btModels = ["Random walk", "Diebold-Li AR(1)", "Gradient boosting"];

  return (
    <div className="page">
      <header>
        <div>
          <h1>MYR Bond Portfolio Risk</h1>
          <p className="sub">MGS curve as of {curve.as_of} · Nelson-Siegel fit error {curve.params.rmse_bp.toFixed(1)}bp</p>
        </div>
        <a className="btn" href={`${API}/factsheet.pdf`} target="_blank">Download factsheet (PDF)</a>
      </header>

      <div className="kpis">
        <Kpi label="Market value" value={`RM${(s.market_value_rm / 1e6).toFixed(1)}m`} />
        <Kpi label="Yield" value={`${s["yield_%"].toFixed(2)}%`} />
        <Kpi label="Modified duration" value={s.mod_duration.toFixed(2)} />
        <Kpi label="Convexity" value={s.convexity.toFixed(1)} />
        <Kpi label="DV01 (per bp)" value={rmPlain(s.dv01_rm)} />
      </div>

      <div className="grid">
        <Card title="Yield curve and OPR shock"
              note="Orange line: the curve after the OPR move, with the long end moving 40% as much as the short end (an assumption).">
          <div className="slider">
            <label>OPR change: <b>{opr > 0 ? "+" : ""}{opr}bp</b></label>
            <input type="range" min={-100} max={100} step={25} value={opr} onChange={(e) => setOpr(Number(e.target.value))} />
          </div>
          <ResponsiveContainer width="100%" height={260}>
            <ComposedChart margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="t" type="number" domain={[0, 11]} tickCount={12} unit="y" />
              <YAxis domain={["auto", "auto"]} unit="%" tickFormatter={(v) => v.toFixed(1)} />
              <Tooltip formatter={(v: number) => `${v.toFixed(3)}%`} labelFormatter={(l) => `${Number(l).toFixed(2)} years`} />
              <Legend />
              <Line data={chartData} dataKey="fit" name="Fitted curve" stroke={NAVY} strokeWidth={2.5} dot={false} />
              <Line data={chartData} dataKey="shocked" name="After OPR move" stroke={ORANGE} strokeDasharray="5 4" dot={false} />
              <Scatter data={marketPts} dataKey="market" name="Market (BNM)" fill="#111" />
            </ComposedChart>
          </ResponsiveContainer>
          {pass && par && (
            <div className="pnl">
              <div>Portfolio P&L (pass-through): <b className={pass.pnl_rm < 0 ? "neg" : "pos"}>{rm(pass.pnl_rm)} ({pass["pnl_%"].toFixed(2)}%)</b></div>
              <div>If the whole curve moved: <b className={par.pnl_rm < 0 ? "neg" : "pos"}>{rm(par.pnl_rm)} ({par["pnl_%"].toFixed(2)}%)</b></div>
            </div>
          )}
        </Card>

        <Card title="OPR stress test (% of portfolio)">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={oprGrid}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="shock" />
              <YAxis unit="%" />
              <Tooltip formatter={(v: number) => `${v.toFixed(2)}%`} />
              <Legend />
              <ReferenceLine y={0} stroke="#111" />
              <Bar dataKey="parallel" name="Parallel shift" fill={NAVY} isAnimationActive={false} />
              <Bar dataKey="OPR pass-through" name="OPR pass-through" fill={ORANGE} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Curve-shape scenarios">
          <table>
            <thead><tr><th>Scenario</th><th className="r">P&L</th><th className="r">%</th></tr></thead>
            <tbody>
              {shapes.map((r) => (
                <tr key={r.scenario}>
                  <td>{r.scenario}</td>
                  <td className={`r ${r.pnl_rm < 0 ? "neg" : "pos"}`}>{rm(r.pnl_rm)}</td>
                  <td className="r">{r["pnl_%"].toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Where the rate risk sits (key-rate DV01)"
              note="Ringgit loss if yields rise 1bp at that maturity only. The bars add up to the total DV01.">
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={krd.map((k) => ({ tenor: `${k.key_tenor_y}y`, v: k.krd_dv01_rm }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="tenor" />
              <YAxis tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
              <Tooltip formatter={(v: number) => rmPlain(v)} />
              <Bar dataKey="v" name="RM per bp">
                {krd.map((_, i) => <Cell key={i} fill={ORANGE} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Holdings">
          <table>
            <thead><tr><th>Holding</th><th>Type</th><th className="r">Yield</th><th className="r">Price</th><th className="r">Weight</th><th className="r">Duration</th></tr></thead>
            <tbody>
              {port.holdings.map((h: Row) => (
                <tr key={h.name}>
                  <td>{h.name}</td><td>{h.type}</td>
                  <td className="r">{h["yield_%"].toFixed(2)}%</td>
                  <td className="r">{h.price.toFixed(2)}</td>
                  <td className="r">{h["weight_%"].toFixed(1)}%</td>
                  <td className="r">{h.mod_duration.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>

        <Card title="Can ML forecast yields? (walk-forward backtest)"
              note={backtest ? `Data: ${backtest.source}. Lower is better. The random walk ("no change") is the benchmark to beat.` : undefined}>
          {backtest ? (
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={backtest.rows.map((r: Row) => ({ ...r, h: `${r.horizon_w}w` }))}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="h" />
                <YAxis unit="bp" />
                <Tooltip formatter={(v: number) => `${v.toFixed(1)}bp`} />
                <Legend />
                {btModels.map((m, i) => <Bar key={m} dataKey={m} fill={[GREY, NAVY, ORANGE][i]} />)}
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="note">Run <code>python scripts/run_backtest.py</code> to populate this panel.</p>
          )}
        </Card>

        <Card title="AI reading of a BNM Monetary Policy Statement"
              note="The AI only reads the statement. Every ringgit figure comes from the risk engine.">
          <textarea placeholder="Paste the full text of a BNM Monetary Policy Statement…" value={mpsText}
                    onChange={(e) => setMpsText(e.target.value)} rows={6} />
          <button className="btn" disabled={!mpsText.trim() || mpsBusy} onClick={runMps}>
            {mpsBusy ? "Reading…" : "Summarise and link to portfolio"}
          </button>
          {mps?.error && <p className="error">{mps.error}</p>}
          {mps?.reading && (
            <div className="mps">
              <p><b>Decision:</b> {mps.reading.opr_decision}{mps.reading.opr_level_pct != null && ` (OPR ${mps.reading.opr_level_pct}%)`} · <b>Stance:</b> {mps.reading.stance} · <b>Bias:</b> {mps.reading.next_move_bias}</p>
              <ul>{mps.reading.key_points.map((k: string) => <li key={k}>{k}</li>)}</ul>
              <p className="impact">{mps.portfolio_impact}</p>
            </div>
          )}
        </Card>
      </div>

      <footer>Illustrative portfolio for demonstration. Yields: Bank Negara Malaysia / Bursa Malaysia Bonds. Not investment advice.</footer>
    </div>
  );
}

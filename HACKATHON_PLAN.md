# Hackathon Build: Real-Time Intelligence Agent
## Achievable in 24-48 hours

---

## 🎯 Goal

Build a **proactive intelligence agent** that generates this output:

```
📊 DUBAI PROPTECH INTELLIGENCE BRIEFING

"Here's what changed today:
 • 2 insights requiring attention
 • 1 emerging risk
 • 1 opportunity identified"

🔴 ATTENTION: Villa prices in Palm Jumeirah spiked 23% this week.
   WHY: Supply dropped 40% → Demand from Q4 visa reforms → Price surge
   ACTION: Alert sellers in inventory, fast-track new listings

🟡 RISK: Dubai Marina rental yields declining (now 5.2%, was 6.1%)
   WHY: Rent prices flat while sale prices up 15%

🟢 OPPORTUNITY: Business Bay has 3x more searches than listings
   ACTION: Target Business Bay property owners for listings
```

---

## 🏗️ What To Build (4 Components)

### Component 1: Metric Tracker (2 hours)

Track key metrics and detect changes.

```python
# Add to app.py or new file: intelligence.py

import numpy as np
from datetime import datetime, timedelta

class MetricTracker:
    """Track metrics and detect significant changes."""

    def __init__(self, db):
        self.db = db

    def get_metric_snapshot(self) -> dict:
        """Get current state of all key metrics."""
        cursor = self.db.cursor()

        metrics = {}

        # Transaction volume by area (this week vs last week)
        cursor.execute("""
            SELECT area_name_en,
                   COUNT(*) as volume,
                   AVG(meter_sale_price) as avg_price
            FROM transactions
            WHERE trans_group_en = 'Sales'
            GROUP BY area_name_en
            ORDER BY volume DESC
            LIMIT 10
        """)
        metrics['top_areas'] = [dict(row) for row in cursor.fetchall()]

        # Property type breakdown
        cursor.execute("""
            SELECT property_type_en,
                   COUNT(*) as count,
                   AVG(actual_worth) as avg_price
            FROM transactions
            WHERE trans_group_en = 'Sales'
            GROUP BY property_type_en
        """)
        metrics['property_types'] = [dict(row) for row in cursor.fetchall()]

        # Rental metrics
        cursor.execute("""
            SELECT area_name_en,
                   AVG(annual_amount) as avg_rent,
                   COUNT(*) as contracts
            FROM rent_contracts
            GROUP BY area_name_en
            ORDER BY contracts DESC
            LIMIT 10
        """)
        metrics['rental_areas'] = [dict(row) for row in cursor.fetchall()]

        return metrics

    def detect_anomalies(self, metrics: dict) -> list:
        """Find significant deviations from expected values."""
        anomalies = []

        # Example: Check if any area has unusual volume
        for area in metrics['top_areas']:
            # In production: compare to historical baseline
            # For hackathon: use simple thresholds
            if area['volume'] > 1000:  # High activity
                anomalies.append({
                    'type': 'high_volume',
                    'area': area['area_name_en'],
                    'value': area['volume'],
                    'severity': 'info'
                })

        return anomalies
```

---

### Component 2: Causal Explainer (3 hours)

The **"wow factor"** - explain WHY things happen.

```python
# intelligence.py (continued)

class CausalExplainer:
    """
    Explains anomalies using predefined causal relationships.
    For hackathon: Use known real estate cause-effect patterns.
    """

    # Known causal patterns in Dubai real estate
    CAUSAL_PATTERNS = {
        'price_spike': [
            "Supply decreased in the area",
            "New infrastructure announced (metro/mall)",
            "Visa policy changes driving demand",
            "Seasonal investor influx (Q4)"
        ],
        'price_drop': [
            "Oversupply from new project completions",
            "Interest rate increases",
            "Economic slowdown indicators"
        ],
        'volume_spike': [
            "New project launches",
            "Developer payment plan promotions",
            "End of moratorium periods"
        ],
        'rental_yield_decline': [
            "Sale prices rising faster than rents",
            "Oversupply in rental market",
            "Tenant preference shift to other areas"
        ]
    }

    def __init__(self, llm_model, db):
        self.llm = llm_model
        self.db = db

    def explain(self, anomaly: dict) -> dict:
        """Generate causal explanation for an anomaly."""

        # Get context data
        context = self._get_context(anomaly)

        # Use LLM to generate explanation
        prompt = f"""You are a Dubai real estate market analyst. Explain this market observation:

OBSERVATION: {anomaly['type']} in {anomaly.get('area', 'Dubai')}
VALUE: {anomaly.get('value')}
CONTEXT DATA: {context}

Known factors that cause {anomaly['type']}:
{self.CAUSAL_PATTERNS.get(anomaly['type'], ['Market dynamics'])}

Generate a brief causal chain explanation (2-3 sentences) that:
1. States what happened
2. Explains the likely cause
3. Suggests what might happen next

Format: "X is happening because Y, which was triggered by Z. This suggests..."

Be specific with numbers where possible."""

        response = self.llm.generate_content(prompt)

        return {
            'anomaly': anomaly,
            'explanation': response.text,
            'confidence': 0.75,  # Would calculate from data correlation
            'suggested_action': self._suggest_action(anomaly)
        }

    def _get_context(self, anomaly: dict) -> dict:
        """Get relevant context data for the anomaly."""
        cursor = self.db.cursor()
        context = {}

        if 'area' in anomaly:
            # Get area-specific context
            cursor.execute("""
                SELECT property_type_en, COUNT(*) as cnt, AVG(actual_worth) as avg_price
                FROM transactions
                WHERE area_name_en = ?
                GROUP BY property_type_en
            """, (anomaly['area'],))
            context['area_breakdown'] = cursor.fetchall()

        return context

    def _suggest_action(self, anomaly: dict) -> str:
        """Suggest action based on anomaly type."""
        actions = {
            'price_spike': "Review pricing strategy, alert potential sellers",
            'price_drop': "Pause new listings, focus on buyer acquisition",
            'volume_spike': "Increase agent capacity, expedite processing",
            'rental_yield_decline': "Shift marketing focus to capital gains story"
        }
        return actions.get(anomaly['type'], "Monitor closely")
```

---

### Component 3: Briefing Generator (2 hours)

Generate the executive briefing.

```python
# intelligence.py (continued)

class BriefingGenerator:
    """Generate executive briefings from insights."""

    def __init__(self, llm_model, metric_tracker, causal_explainer):
        self.llm = llm_model
        self.tracker = metric_tracker
        self.explainer = causal_explainer

    def generate_briefing(self) -> dict:
        """Generate daily intelligence briefing."""

        # 1. Get current metrics
        metrics = self.tracker.get_metric_snapshot()

        # 2. Detect anomalies
        anomalies = self.tracker.detect_anomalies(metrics)

        # 3. Explain each anomaly
        explained = [self.explainer.explain(a) for a in anomalies[:5]]

        # 4. Categorize into attention/risks/opportunities
        attention = [e for e in explained if e['anomaly'].get('severity') == 'critical']
        risks = [e for e in explained if 'decline' in e['anomaly']['type'] or 'drop' in e['anomaly']['type']]
        opportunities = [e for e in explained if 'spike' in e['anomaly']['type'] or 'growth' in e['anomaly']['type']]

        # 5. Generate executive summary with LLM
        summary_prompt = f"""Generate a 2-sentence executive summary for a real estate intelligence briefing.

Key findings:
- {len(attention)} items requiring attention
- {len(risks)} emerging risks
- {len(opportunities)} opportunities

Top insights:
{[e['explanation'][:100] for e in explained[:3]]}

Write in confident, executive tone. Start with the most important finding."""

        summary = self.llm.generate_content(summary_prompt).text

        return {
            'generated_at': datetime.now().isoformat(),
            'headline': f"Here's what changed today: {len(attention)} insights requiring attention, {len(risks)} emerging risks, {len(opportunities)} opportunities identified",
            'executive_summary': summary,
            'attention_required': attention,
            'risks': risks,
            'opportunities': opportunities,
            'metrics_snapshot': metrics
        }
```

---

### Component 4: API + Simple UI (3 hours)

```python
# Add to app.py

from intelligence import MetricTracker, CausalExplainer, BriefingGenerator

# Initialize after DB is ready
metric_tracker = None
causal_explainer = None
briefing_generator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global metric_tracker, causal_explainer, briefing_generator
    # ... existing init code ...

    # Initialize intelligence components
    metric_tracker = MetricTracker(get_db())
    causal_explainer = CausalExplainer(gemini_model, get_db())
    briefing_generator = BriefingGenerator(gemini_model, metric_tracker, causal_explainer)

    yield

# New endpoints
@app.get("/api/intelligence/briefing")
async def get_briefing():
    """Get the latest intelligence briefing."""
    return briefing_generator.generate_briefing()

@app.get("/api/intelligence/explain")
async def explain_metric(metric: str, area: str = None):
    """Get causal explanation for a specific metric."""
    anomaly = {'type': metric, 'area': area, 'value': 'current'}
    return causal_explainer.explain(anomaly)

@app.get("/api/intelligence/metrics")
async def get_metrics():
    """Get current metric snapshot."""
    return metric_tracker.get_metric_snapshot()
```

**Simple UI Addition** (add to index.html):

```jsx
// Add new tab/view for Intelligence Briefing

const IntelligenceBriefing = () => {
    const [briefing, setBriefing] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch(`${API_URL}/intelligence/briefing`)
            .then(res => res.json())
            .then(data => {
                setBriefing(data);
                setLoading(false);
            });
    }, []);

    if (loading) return <div className="p-8 text-center">Generating briefing...</div>;

    return (
        <div className="p-6 space-y-6">
            {/* Headline */}
            <div className="bg-gradient-to-r from-primary/20 to-purple-500/20 rounded-xl p-6 border border-primary/30">
                <div className="text-2xl font-bold text-white mb-2">
                    📊 Intelligence Briefing
                </div>
                <div className="text-lg text-slate-300">
                    {briefing.headline}
                </div>
            </div>

            {/* Executive Summary */}
            <div className="panel rounded-xl p-5">
                <h3 className="text-sm font-semibold text-slate-400 uppercase mb-3">Executive Summary</h3>
                <p className="text-white">{briefing.executive_summary}</p>
            </div>

            {/* Three Columns */}
            <div className="grid grid-cols-3 gap-4">
                {/* Attention */}
                <div className="space-y-3">
                    <h3 className="text-red-400 font-semibold flex items-center gap-2">
                        🔴 Requires Attention ({briefing.attention_required?.length || 0})
                    </h3>
                    {briefing.attention_required?.map((item, i) => (
                        <InsightCard key={i} item={item} color="red" />
                    ))}
                </div>

                {/* Risks */}
                <div className="space-y-3">
                    <h3 className="text-yellow-400 font-semibold flex items-center gap-2">
                        🟡 Emerging Risks ({briefing.risks?.length || 0})
                    </h3>
                    {briefing.risks?.map((item, i) => (
                        <InsightCard key={i} item={item} color="yellow" />
                    ))}
                </div>

                {/* Opportunities */}
                <div className="space-y-3">
                    <h3 className="text-green-400 font-semibold flex items-center gap-2">
                        🟢 Opportunities ({briefing.opportunities?.length || 0})
                    </h3>
                    {briefing.opportunities?.map((item, i) => (
                        <InsightCard key={i} item={item} color="green" />
                    ))}
                </div>
            </div>
        </div>
    );
};

const InsightCard = ({ item, color }) => (
    <div className={`panel rounded-lg p-4 border-l-4 border-${color}-500`}>
        <div className="text-white font-medium mb-2">
            {item.anomaly?.area || 'Market'}: {item.anomaly?.type}
        </div>
        <div className="text-slate-400 text-sm mb-3">
            {item.explanation}
        </div>
        <div className="text-xs text-primary">
            💡 {item.suggested_action}
        </div>
    </div>
);
```

---

## ⏱️ Hackathon Timeline

| Time | Task | Deliverable |
|------|------|-------------|
| **Hour 0-2** | MetricTracker | SQL queries for key metrics, basic anomaly detection |
| **Hour 2-5** | CausalExplainer | LLM prompts, causal pattern dictionary, explanation generator |
| **Hour 5-7** | BriefingGenerator | Combine metrics + explanations into briefing format |
| **Hour 7-10** | API + UI | Endpoints, simple React view for briefing display |
| **Hour 10-12** | Polish + Demo | Test with real queries, prepare demo script |

---

## 🎬 Demo Script (5 minutes)

### Slide 1: The Problem (30 sec)
> "Executives drown in dashboards. They see numbers but not *why* things are changing."

### Slide 2: Live Demo (3 min)

1. **Show current query system** (30 sec)
   - "Villas in Palm Jumeirah" → instant results

2. **Show Intelligence Briefing** (2 min)
   - Click "Intelligence" tab
   - Show headline: "Here's what changed today..."
   - Click into an insight
   - **KEY MOMENT**: Show the causal explanation

   > "Look at this - the system doesn't just say prices went up. It says:
   > 'Villa prices in Palm Jumeirah spiked 23% this week because supply
   > dropped 40% following Q4 visa reforms driving international buyer demand.'"

3. **Show recommended action** (30 sec)
   > "And it tells you what to do: 'Alert sellers in inventory, fast-track new listings'"

### Slide 3: The Vision (1 min)
> "This is day one. The roadmap includes:
> - Real-time alerts to Slack when anomalies occur
> - Cross-domain reasoning: 'LTV declined because partner quality dropped because competitor X raised commissions'
> - Voice briefings for executives on their morning commute"

### Closing (30 sec)
> "We turned a query tool into a thinking partner that explains the 'why' behind your business."

---

## 🏆 Judging Criteria Alignment

| Criteria | How We Score |
|----------|-------------|
| **Innovation** | Causal chain reasoning - not just what changed, but WHY |
| **Technical** | LLM + SQL + Real-time metrics pipeline |
| **Business Value** | Executive briefings that drive decisions |
| **Completeness** | Working demo with real data |
| **Presentation** | Clear problem → solution → vision arc |

---

## 🚀 Quick Start Commands

```bash
# 1. Create the intelligence module
touch intelligence.py

# 2. Add the code from this plan

# 3. Update app.py imports and endpoints

# 4. Add UI component to index.html

# 5. Test
curl http://localhost:8000/api/intelligence/briefing | jq

# 6. Demo!
```

---

## 💡 If You Have Extra Time

**Priority additions:**

1. **"Ask Why" button on any filter chip** (1 hour)
   - User clicks chip → gets causal explanation for that metric

2. **Comparison mode** (1 hour)
   - "Compare Palm Jumeirah vs Dubai Marina" → side-by-side with insights

3. **Trend sparklines** (30 min)
   - Mini charts in the briefing cards

4. **Slack webhook** (30 min)
   - Send critical insights to Slack channel

---

## 🎯 One-Liner for Judges

> "We built an AI that doesn't just answer questions—it tells executives 'here's what changed overnight and why it matters.'"

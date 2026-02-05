# Real-Time Enterprise Intelligence Agent
## Architecture & Implementation Plan

---

## 🎯 Vision

Transform the current query-response system into a **proactive intelligence agent** that:
- Continuously monitors cross-domain data streams
- Detects anomalies, trends, and causal relationships
- Generates executive briefings with actionable insights
- Explains the "why" behind business changes through causal chain reasoning

---

## 📐 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ENTERPRISE INTELLIGENCE AGENT                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   INGEST     │  │   PROCESS    │  │   REASON     │  │   DELIVER    │    │
│  │   LAYER      │──▶│   LAYER      │──▶│   LAYER      │──▶│   LAYER      │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│        │                  │                  │                  │           │
│        ▼                  ▼                  ▼                  ▼           │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        KNOWLEDGE GRAPH + VECTOR STORE                 │  │
│  │            (Unified Business Context & Historical Memory)             │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Phase 1: Multi-Domain Data Unification (Weeks 1-3)

### 1.1 Data Domain Registry

```python
# New file: domains/domain_registry.py

BUSINESS_DOMAINS = {
    "real_estate": {
        "tables": ["transactions", "rent_contracts", "units", "developers", "projects"],
        "metrics": ["transaction_volume", "avg_price", "rental_yield", "occupancy_rate"],
        "dimensions": ["area", "property_type", "transaction_type", "time"],
        "refresh_interval": "5min"
    },
    "customer": {
        "tables": ["clients", "client_transactions", "client_interactions"],
        "metrics": ["ltv", "acquisition_cost", "churn_rate", "nps_score"],
        "dimensions": ["segment", "channel", "tenure", "product_mix"],
        "refresh_interval": "15min"
    },
    "partner": {
        "tables": ["partners", "partner_deals", "commissions", "partner_performance"],
        "metrics": ["deal_volume", "commission_rate", "conversion_rate", "quality_score"],
        "dimensions": ["partner_tier", "channel_type", "region"],
        "refresh_interval": "1hour"
    },
    "market": {
        "tables": ["competitor_pricing", "market_indices", "economic_indicators"],
        "metrics": ["market_share", "price_index", "supply_demand_ratio"],
        "dimensions": ["segment", "geography", "time"],
        "refresh_interval": "1day"
    },
    "operations": {
        "tables": ["service_requests", "response_times", "escalations"],
        "metrics": ["resolution_time", "satisfaction_score", "escalation_rate"],
        "dimensions": ["request_type", "priority", "team"],
        "refresh_interval": "5min"
    }
}
```

### 1.2 Cross-Domain Relationship Map

```python
# New file: domains/relationship_graph.py

CROSS_DOMAIN_LINKS = {
    # Customer LTV is influenced by:
    ("customer.ltv", "partner.quality_score"): {
        "relationship": "positive_correlation",
        "lag": "30days",
        "strength": 0.72
    },
    ("customer.ltv", "real_estate.rental_yield"): {
        "relationship": "positive_correlation",
        "lag": "0days",
        "strength": 0.65
    },

    # Partner quality is influenced by:
    ("partner.quality_score", "partner.commission_rate"): {
        "relationship": "threshold_effect",  # Below threshold = low quality partners
        "threshold": "market_average",
        "lag": "90days"
    },
    ("partner.commission_rate", "market.competitor_commission"): {
        "relationship": "competitive_pressure",
        "lag": "30days"
    },

    # Market dynamics:
    ("real_estate.transaction_volume", "market.supply_demand_ratio"): {
        "relationship": "inverse_correlation",
        "lag": "14days"
    }
}
```

### 1.3 Unified Data Lake Schema

```sql
-- New tables for cross-domain analytics

CREATE TABLE metric_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL,
    dimensions JSON,  -- {"area": "Palm Jumeirah", "property_type": "Villa"}
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    period_type TEXT,  -- 'hourly', 'daily', 'weekly'
    INDEX idx_domain_metric_time (domain, metric_name, timestamp)
);

CREATE TABLE causal_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT,  -- 'anomaly', 'trend_change', 'threshold_breach'
    source_domain TEXT,
    source_metric TEXT,
    magnitude REAL,  -- How significant (z-score or % change)
    detected_at DATETIME,
    root_cause_chain JSON,  -- Linked causal explanations
    confidence REAL,
    status TEXT  -- 'new', 'acknowledged', 'resolved'
);

CREATE TABLE insight_briefings (
    briefing_id TEXT PRIMARY KEY,
    briefing_type TEXT,  -- 'daily', 'alert', 'weekly'
    generated_at DATETIME,
    insights JSON,  -- Array of insight objects
    risks JSON,
    opportunities JSON,
    executive_summary TEXT,
    detailed_analysis TEXT
);
```

---

## 🧠 Phase 2: Intelligence Engine (Weeks 4-7)

### 2.1 Anomaly Detection System

```python
# New file: intelligence/anomaly_detector.py

from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import numpy as np
from scipy import stats

@dataclass
class Anomaly:
    metric: str
    domain: str
    current_value: float
    expected_value: float
    deviation: float  # z-score
    direction: str  # 'increase' or 'decrease'
    severity: str  # 'info', 'warning', 'critical'
    detected_at: datetime
    context: Dict  # Relevant dimensions

class AnomalyDetector:
    """Multi-method anomaly detection for business metrics."""

    def __init__(self, db_connection, lookback_days: int = 90):
        self.db = db_connection
        self.lookback_days = lookback_days
        self.detection_methods = [
            self._zscore_detection,
            self._iqr_detection,
            self._trend_deviation,
            self._seasonal_deviation,
        ]

    def detect_anomalies(self, domain: str, metric: str) -> List[Anomaly]:
        """Run all detection methods and aggregate results."""
        historical = self._get_historical_data(domain, metric)
        current = self._get_current_value(domain, metric)

        anomalies = []
        for method in self.detection_methods:
            result = method(historical, current)
            if result:
                anomalies.append(result)

        # Consensus scoring - more methods agree = higher confidence
        return self._aggregate_anomalies(anomalies)

    def _zscore_detection(self, historical: np.array, current: float) -> Optional[Anomaly]:
        """Statistical z-score based detection."""
        mean = np.mean(historical)
        std = np.std(historical)

        if std == 0:
            return None

        z_score = (current - mean) / std

        if abs(z_score) > 2.5:
            return Anomaly(
                deviation=z_score,
                severity='critical' if abs(z_score) > 3.5 else 'warning',
                direction='increase' if z_score > 0 else 'decrease',
                expected_value=mean,
                current_value=current
            )
        return None

    def _trend_deviation(self, historical: np.array, current: float) -> Optional[Anomaly]:
        """Detect deviation from established trend."""
        # Fit linear trend
        x = np.arange(len(historical))
        slope, intercept, r_value, _, _ = stats.linregress(x, historical)

        # Project expected value
        expected = slope * len(historical) + intercept

        # Calculate residual std from trend
        residuals = historical - (slope * x + intercept)
        residual_std = np.std(residuals)

        deviation = (current - expected) / residual_std if residual_std > 0 else 0

        if abs(deviation) > 2.0:
            return Anomaly(
                deviation=deviation,
                severity='warning',
                direction='increase' if deviation > 0 else 'decrease',
                expected_value=expected,
                current_value=current
            )
        return None
```

### 2.2 Causal Chain Reasoner

```python
# New file: intelligence/causal_reasoner.py

from typing import List, Dict, Tuple
from dataclasses import dataclass
import networkx as nx

@dataclass
class CausalLink:
    cause_metric: str
    cause_domain: str
    effect_metric: str
    effect_domain: str
    relationship: str
    confidence: float
    time_lag: int  # days
    evidence: str  # Natural language explanation

@dataclass
class CausalChain:
    """A chain of causally linked events explaining a business change."""
    root_cause: str
    chain: List[CausalLink]
    final_effect: str
    overall_confidence: float
    narrative: str  # Human-readable explanation

class CausalReasoner:
    """
    Builds causal explanations by traversing the business relationship graph.

    Example output:
    "Client LTV is declining (↓12%) because partner mix shifted toward
    lower-quality channels (quality score ↓18%), which happened because
    our commission structure became uncompetitive (↓5% vs market) after
    competitor X raised partner payouts (+8% announced 90 days ago)."
    """

    def __init__(self, relationship_graph: Dict, metric_history: Dict):
        self.graph = self._build_causal_graph(relationship_graph)
        self.metrics = metric_history

    def _build_causal_graph(self, relationships: Dict) -> nx.DiGraph:
        """Build directed graph of causal relationships."""
        G = nx.DiGraph()

        for (cause, effect), attrs in relationships.items():
            G.add_edge(cause, effect, **attrs)

        return G

    def explain_anomaly(self, anomaly: 'Anomaly', max_depth: int = 4) -> CausalChain:
        """
        Given an anomaly, trace back through causal graph to find root cause.
        Uses BFS with time-lag awareness.
        """
        target = f"{anomaly.domain}.{anomaly.metric}"

        # Find all paths leading to this metric
        causal_paths = []

        for node in self.graph.nodes():
            if node == target:
                continue

            try:
                paths = list(nx.all_simple_paths(
                    self.graph, node, target, cutoff=max_depth
                ))
                for path in paths:
                    chain = self._evaluate_path(path, anomaly)
                    if chain and chain.overall_confidence > 0.5:
                        causal_paths.append(chain)
            except nx.NetworkXNoPath:
                continue

        # Rank by confidence and return best explanation
        if causal_paths:
            best = max(causal_paths, key=lambda c: c.overall_confidence)
            best.narrative = self._generate_narrative(best)
            return best

        return None

    def _evaluate_path(self, path: List[str], anomaly: 'Anomaly') -> CausalChain:
        """Evaluate if a causal path explains the observed anomaly."""
        links = []
        cumulative_lag = 0
        confidence = 1.0

        for i in range(len(path) - 1):
            cause, effect = path[i], path[i+1]
            edge_data = self.graph.edges[cause, effect]

            # Check if cause metric also changed in expected direction
            cause_domain, cause_metric = cause.split('.')
            cause_change = self._get_metric_change(
                cause_domain, cause_metric,
                days_ago=cumulative_lag + edge_data.get('lag', 0)
            )

            if not self._changes_align(cause_change, edge_data['relationship']):
                confidence *= 0.5  # Penalize misaligned changes

            links.append(CausalLink(
                cause_metric=cause_metric,
                cause_domain=cause_domain,
                effect_metric=effect.split('.')[1],
                effect_domain=effect.split('.')[0],
                relationship=edge_data['relationship'],
                confidence=edge_data.get('strength', 0.5),
                time_lag=edge_data.get('lag', 0),
                evidence=f"{cause_metric} changed by {cause_change:+.1%}"
            ))

            cumulative_lag += edge_data.get('lag', 0)
            confidence *= edge_data.get('strength', 0.5)

        return CausalChain(
            root_cause=path[0],
            chain=links,
            final_effect=path[-1],
            overall_confidence=confidence,
            narrative=""
        )

    def _generate_narrative(self, chain: CausalChain) -> str:
        """Generate human-readable causal explanation."""
        parts = []

        # Start with the observed effect
        final_link = chain.chain[-1]
        parts.append(
            f"{final_link.effect_metric.replace('_', ' ').title()} is "
            f"{'declining' if 'decrease' in str(chain) else 'changing'}"
        )

        # Add each causal link
        for i, link in enumerate(reversed(chain.chain)):
            connector = "because" if i == 0 else "which happened because"
            parts.append(
                f"{connector} {link.cause_metric.replace('_', ' ')} "
                f"({link.evidence})"
            )

        return " ".join(parts) + "."
```

### 2.3 Cross-Domain Insight Generator

```python
# New file: intelligence/insight_generator.py

from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class InsightType(Enum):
    ANOMALY = "anomaly"
    TREND = "trend"
    CORRELATION = "correlation"
    OPPORTUNITY = "opportunity"
    RISK = "risk"

class InsightPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class Insight:
    insight_id: str
    insight_type: InsightType
    priority: InsightPriority
    title: str
    summary: str
    detailed_analysis: str
    affected_domains: List[str]
    metrics_involved: List[str]
    causal_chain: Optional['CausalChain']
    recommended_actions: List[str]
    confidence: float
    generated_at: datetime

class InsightGenerator:
    """
    Generates actionable insights by combining:
    - Anomaly detection results
    - Causal reasoning
    - Cross-domain correlation analysis
    - Trend analysis
    """

    def __init__(self, anomaly_detector, causal_reasoner, db_connection):
        self.anomaly_detector = anomaly_detector
        self.causal_reasoner = causal_reasoner
        self.db = db_connection

        # Insight templates for different scenarios
        self.templates = {
            'ltv_decline': self._template_ltv_decline,
            'volume_spike': self._template_volume_spike,
            'margin_compression': self._template_margin_compression,
            'churn_risk': self._template_churn_risk,
            'market_opportunity': self._template_market_opportunity,
        }

    def generate_daily_insights(self) -> List[Insight]:
        """Generate all insights for daily briefing."""
        insights = []

        # 1. Detect anomalies across all domains
        for domain, config in BUSINESS_DOMAINS.items():
            for metric in config['metrics']:
                anomalies = self.anomaly_detector.detect_anomalies(domain, metric)

                for anomaly in anomalies:
                    # Get causal explanation
                    causal_chain = self.causal_reasoner.explain_anomaly(anomaly)

                    insight = self._anomaly_to_insight(anomaly, causal_chain)
                    insights.append(insight)

        # 2. Check for cross-domain correlations
        correlations = self._detect_cross_domain_correlations()
        insights.extend(correlations)

        # 3. Identify opportunities
        opportunities = self._identify_opportunities()
        insights.extend(opportunities)

        # 4. Assess risks
        risks = self._assess_risks()
        insights.extend(risks)

        # Deduplicate and prioritize
        return self._prioritize_insights(insights)

    def _anomaly_to_insight(self, anomaly: 'Anomaly', causal_chain: 'CausalChain') -> Insight:
        """Convert detected anomaly into actionable insight."""

        # Determine insight type and priority
        if anomaly.severity == 'critical':
            priority = InsightPriority.CRITICAL
        elif anomaly.severity == 'warning':
            priority = InsightPriority.HIGH
        else:
            priority = InsightPriority.MEDIUM

        # Generate title
        direction = "increased" if anomaly.direction == 'increase' else "decreased"
        title = f"{anomaly.metric.replace('_', ' ').title()} {direction} significantly"

        # Generate summary with causal explanation
        if causal_chain:
            summary = causal_chain.narrative
        else:
            summary = (
                f"{anomaly.metric} in {anomaly.domain} has {direction} by "
                f"{abs(anomaly.deviation):.1f} standard deviations from expected. "
                f"Current: {anomaly.current_value:,.0f}, Expected: {anomaly.expected_value:,.0f}"
            )

        # Generate recommended actions
        actions = self._generate_recommendations(anomaly, causal_chain)

        return Insight(
            insight_id=f"INS-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            insight_type=InsightType.ANOMALY,
            priority=priority,
            title=title,
            summary=summary,
            detailed_analysis=self._generate_detailed_analysis(anomaly, causal_chain),
            affected_domains=[anomaly.domain] + (
                [link.cause_domain for link in causal_chain.chain] if causal_chain else []
            ),
            metrics_involved=[anomaly.metric],
            causal_chain=causal_chain,
            recommended_actions=actions,
            confidence=causal_chain.overall_confidence if causal_chain else 0.6,
            generated_at=datetime.now()
        )
```

---

## 📊 Phase 3: Executive Briefing System (Weeks 8-10)

### 3.1 Briefing Generator

```python
# New file: briefings/briefing_generator.py

from typing import List
from dataclasses import dataclass
from datetime import datetime
import google.generativeai as genai

@dataclass
class ExecutiveBriefing:
    briefing_id: str
    generated_at: datetime
    time_period: str  # "overnight", "weekly", "monthly"

    # Structured sections
    headline: str
    executive_summary: str

    attention_required: List[Insight]  # Critical items
    emerging_risks: List[Insight]
    opportunities: List[Insight]

    key_metrics_snapshot: Dict[str, Dict]
    trend_summary: str

    # For detailed drill-down
    full_analysis: str
    data_appendix: Dict

class BriefingGenerator:
    """
    Generates executive briefings with the format:

    "Here's what changed overnight:
     - 3 insights requiring attention
     - 2 emerging risks
     - 1 opportunity identified"
    """

    def __init__(self, insight_generator, llm_model):
        self.insight_gen = insight_generator
        self.llm = llm_model

    def generate_morning_briefing(self) -> ExecutiveBriefing:
        """Generate overnight briefing for executives."""

        # Get all insights from last 24 hours
        insights = self.insight_gen.generate_daily_insights()

        # Categorize
        attention_items = [i for i in insights if i.priority in [InsightPriority.CRITICAL, InsightPriority.HIGH]]
        risks = [i for i in insights if i.insight_type == InsightType.RISK]
        opportunities = [i for i in insights if i.insight_type == InsightType.OPPORTUNITY]

        # Generate headline
        headline = self._generate_headline(attention_items, risks, opportunities)

        # Generate executive summary using LLM
        exec_summary = self._generate_executive_summary(insights)

        # Get key metrics snapshot
        metrics_snapshot = self._get_metrics_snapshot()

        return ExecutiveBriefing(
            briefing_id=f"BRIEF-{datetime.now().strftime('%Y%m%d')}",
            generated_at=datetime.now(),
            time_period="overnight",
            headline=headline,
            executive_summary=exec_summary,
            attention_required=attention_items[:5],
            emerging_risks=risks[:3],
            opportunities=opportunities[:3],
            key_metrics_snapshot=metrics_snapshot,
            trend_summary=self._generate_trend_summary(),
            full_analysis=self._generate_full_analysis(insights),
            data_appendix={}
        )

    def _generate_headline(self, attention, risks, opportunities) -> str:
        """Generate the briefing headline."""
        parts = []

        if attention:
            parts.append(f"{len(attention)} insight{'s' if len(attention) > 1 else ''} requiring attention")
        if risks:
            parts.append(f"{len(risks)} emerging risk{'s' if len(risks) > 1 else ''}")
        if opportunities:
            parts.append(f"{len(opportunities)} opportunit{'ies' if len(opportunities) > 1 else 'y'} identified")

        if not parts:
            return "All systems nominal - no significant changes overnight"

        return "Here's what changed overnight: " + ", ".join(parts)

    def _generate_executive_summary(self, insights: List[Insight]) -> str:
        """Use LLM to generate natural language executive summary."""

        # Prepare context for LLM
        insight_summaries = "\n".join([
            f"- [{i.priority.name}] {i.title}: {i.summary}"
            for i in insights[:10]
        ])

        prompt = f"""Generate a concise executive briefing summary (3-4 sentences) from these business insights:

{insight_summaries}

Requirements:
- Lead with the most important finding
- Use specific numbers where available
- Explain causal relationships clearly
- End with recommended focus area for today

Write in a direct, executive communication style."""

        response = self.llm.generate_content(prompt)
        return response.text
```

### 3.2 Real-Time Alert System

```python
# New file: alerts/alert_manager.py

from typing import List, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio

class AlertChannel(Enum):
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    DASHBOARD = "dashboard"
    WEBHOOK = "webhook"

@dataclass
class AlertRule:
    rule_id: str
    name: str
    domain: str
    metric: str
    condition: str  # "value > threshold", "change > 10%", "anomaly_score > 2.5"
    threshold: float
    channels: List[AlertChannel]
    recipients: List[str]
    cooldown_minutes: int  # Prevent alert fatigue
    enabled: bool

@dataclass
class Alert:
    alert_id: str
    rule: AlertRule
    triggered_at: datetime
    current_value: float
    threshold_value: float
    insight: 'Insight'
    status: str  # 'fired', 'acknowledged', 'resolved'

class AlertManager:
    """
    Real-time alerting system with:
    - Configurable rules per metric/domain
    - Multi-channel delivery (Slack, email, SMS)
    - Cooldown periods to prevent alert fatigue
    - Acknowledgment and resolution tracking
    """

    def __init__(self, db_connection, insight_generator):
        self.db = db_connection
        self.insight_gen = insight_generator
        self.rules: List[AlertRule] = self._load_rules()
        self.recent_alerts: Dict[str, datetime] = {}  # For cooldown tracking

        # Channel handlers
        self.channel_handlers: Dict[AlertChannel, Callable] = {
            AlertChannel.SLACK: self._send_slack,
            AlertChannel.EMAIL: self._send_email,
            AlertChannel.DASHBOARD: self._push_to_dashboard,
            AlertChannel.WEBHOOK: self._send_webhook,
        }

    async def monitor_loop(self):
        """Continuous monitoring loop."""
        while True:
            await self._check_all_rules()
            await asyncio.sleep(60)  # Check every minute

    async def _check_all_rules(self):
        """Evaluate all alert rules."""
        for rule in self.rules:
            if not rule.enabled:
                continue

            # Check cooldown
            last_fired = self.recent_alerts.get(rule.rule_id)
            if last_fired:
                minutes_since = (datetime.now() - last_fired).seconds / 60
                if minutes_since < rule.cooldown_minutes:
                    continue

            # Evaluate rule
            current_value = self._get_current_value(rule.domain, rule.metric)
            should_fire = self._evaluate_condition(rule.condition, current_value, rule.threshold)

            if should_fire:
                await self._fire_alert(rule, current_value)

    async def _fire_alert(self, rule: AlertRule, current_value: float):
        """Fire an alert through all configured channels."""

        # Generate insight for context
        insight = self.insight_gen.generate_for_metric(rule.domain, rule.metric)

        alert = Alert(
            alert_id=f"ALERT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            rule=rule,
            triggered_at=datetime.now(),
            current_value=current_value,
            threshold_value=rule.threshold,
            insight=insight,
            status='fired'
        )

        # Send to all channels
        for channel in rule.channels:
            handler = self.channel_handlers.get(channel)
            if handler:
                await handler(alert, rule.recipients)

        # Record for cooldown
        self.recent_alerts[rule.rule_id] = datetime.now()

        # Store in database
        self._store_alert(alert)

    async def _send_slack(self, alert: Alert, recipients: List[str]):
        """Send alert to Slack."""
        message = self._format_slack_message(alert)
        # Slack API integration here

    def _format_slack_message(self, alert: Alert) -> Dict:
        """Format alert as rich Slack message."""
        return {
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🚨 {alert.rule.name}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*{alert.insight.title}*\n{alert.insight.summary}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Current:* {alert.current_value:,.0f}"},
                        {"type": "mrkdwn", "text": f"*Threshold:* {alert.threshold_value:,.0f}"},
                        {"type": "mrkdwn", "text": f"*Domain:* {alert.rule.domain}"},
                        {"type": "mrkdwn", "text": f"*Metric:* {alert.rule.metric}"}
                    ]
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"Confidence: {alert.insight.confidence:.0%}"}
                    ]
                }
            ]
        }
```

---

## 🖥️ Phase 4: Enhanced UI (Weeks 11-13)

### 4.1 Executive Dashboard Component

```jsx
// New file: components/ExecutiveDashboard.jsx

const ExecutiveDashboard = () => {
    const [briefing, setBriefing] = useState(null);
    const [selectedInsight, setSelectedInsight] = useState(null);

    useEffect(() => {
        // Fetch morning briefing
        fetch('/api/briefings/latest')
            .then(res => res.json())
            .then(setBriefing);

        // Subscribe to real-time alerts
        const ws = new WebSocket('ws://localhost:8000/ws/alerts');
        ws.onmessage = (event) => {
            const alert = JSON.parse(event.data);
            showAlertNotification(alert);
        };

        return () => ws.close();
    }, []);

    return (
        <div className="executive-dashboard">
            {/* Headline Banner */}
            <div className="headline-banner">
                <div className="headline-icon">
                    {briefing?.attention_required.length > 0 ? '⚠️' : '✅'}
                </div>
                <h1>{briefing?.headline}</h1>
                <span className="timestamp">
                    Generated {formatTime(briefing?.generated_at)}
                </span>
            </div>

            {/* Executive Summary */}
            <div className="exec-summary panel">
                <h2>Executive Summary</h2>
                <p>{briefing?.executive_summary}</p>
            </div>

            {/* Three Column Layout */}
            <div className="insights-grid">
                {/* Attention Required */}
                <div className="insight-column critical">
                    <h3>🔴 Requires Attention ({briefing?.attention_required.length})</h3>
                    {briefing?.attention_required.map(insight => (
                        <InsightCard
                            key={insight.insight_id}
                            insight={insight}
                            onClick={() => setSelectedInsight(insight)}
                        />
                    ))}
                </div>

                {/* Emerging Risks */}
                <div className="insight-column warning">
                    <h3>🟡 Emerging Risks ({briefing?.emerging_risks.length})</h3>
                    {briefing?.emerging_risks.map(insight => (
                        <InsightCard
                            key={insight.insight_id}
                            insight={insight}
                            onClick={() => setSelectedInsight(insight)}
                        />
                    ))}
                </div>

                {/* Opportunities */}
                <div className="insight-column success">
                    <h3>🟢 Opportunities ({briefing?.opportunities.length})</h3>
                    {briefing?.opportunities.map(insight => (
                        <InsightCard
                            key={insight.insight_id}
                            insight={insight}
                            onClick={() => setSelectedInsight(insight)}
                        />
                    ))}
                </div>
            </div>

            {/* Insight Detail Modal */}
            {selectedInsight && (
                <InsightDetailModal
                    insight={selectedInsight}
                    onClose={() => setSelectedInsight(null)}
                />
            )}
        </div>
    );
};

const InsightCard = ({ insight, onClick }) => (
    <div className="insight-card" onClick={onClick}>
        <div className="insight-header">
            <span className={`priority-badge ${insight.priority}`}>
                {insight.priority}
            </span>
            <span className="confidence">
                {(insight.confidence * 100).toFixed(0)}% confident
            </span>
        </div>
        <h4>{insight.title}</h4>
        <p>{insight.summary}</p>
        <div className="affected-domains">
            {insight.affected_domains.map(d => (
                <span key={d} className="domain-tag">{d}</span>
            ))}
        </div>
    </div>
);

const InsightDetailModal = ({ insight, onClose }) => (
    <div className="modal-overlay" onClick={onClose}>
        <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h2>{insight.title}</h2>

            {/* Causal Chain Visualization */}
            {insight.causal_chain && (
                <div className="causal-chain">
                    <h3>Root Cause Analysis</h3>
                    <CausalChainDiagram chain={insight.causal_chain} />
                    <p className="narrative">{insight.causal_chain.narrative}</p>
                </div>
            )}

            {/* Detailed Analysis */}
            <div className="detailed-analysis">
                <h3>Detailed Analysis</h3>
                <p>{insight.detailed_analysis}</p>
            </div>

            {/* Recommended Actions */}
            <div className="recommendations">
                <h3>Recommended Actions</h3>
                <ul>
                    {insight.recommended_actions.map((action, i) => (
                        <li key={i}>{action}</li>
                    ))}
                </ul>
            </div>
        </div>
    </div>
);
```

### 4.2 Causal Chain Visualization

```jsx
// Visual representation of causal relationships

const CausalChainDiagram = ({ chain }) => {
    return (
        <div className="causal-diagram">
            {chain.chain.map((link, i) => (
                <React.Fragment key={i}>
                    {/* Cause Node */}
                    <div className={`causal-node ${i === 0 ? 'root' : ''}`}>
                        <div className="domain-label">{link.cause_domain}</div>
                        <div className="metric-name">{link.cause_metric}</div>
                        <div className="evidence">{link.evidence}</div>
                    </div>

                    {/* Arrow with relationship */}
                    <div className="causal-arrow">
                        <div className="arrow-line">→</div>
                        <div className="relationship-label">
                            {link.relationship}
                            <span className="lag">({link.time_lag}d lag)</span>
                        </div>
                        <div className="confidence-bar"
                             style={{width: `${link.confidence * 100}%`}} />
                    </div>
                </React.Fragment>
            ))}

            {/* Final Effect Node */}
            <div className="causal-node effect">
                <div className="domain-label">{chain.chain[chain.chain.length-1].effect_domain}</div>
                <div className="metric-name">{chain.final_effect.split('.')[1]}</div>
                <div className="impact">OBSERVED CHANGE</div>
            </div>
        </div>
    );
};
```

---

## 🔌 Phase 5: API & Integration Layer (Weeks 14-16)

### 5.1 New API Endpoints

```python
# Add to app.py

# ============================================================================
# Intelligence & Briefing Endpoints
# ============================================================================

@app.get("/api/briefings/latest")
async def get_latest_briefing():
    """Get the most recent executive briefing."""
    generator = BriefingGenerator(insight_gen, gemini_model)
    return generator.generate_morning_briefing()

@app.get("/api/briefings/{briefing_id}")
async def get_briefing(briefing_id: str):
    """Get a specific briefing by ID."""
    # Fetch from database
    pass

@app.get("/api/insights")
async def get_insights(
    domain: Optional[str] = None,
    priority: Optional[str] = None,
    limit: int = 20
):
    """Get filtered list of insights."""
    insights = insight_gen.generate_daily_insights()

    if domain:
        insights = [i for i in insights if domain in i.affected_domains]
    if priority:
        insights = [i for i in insights if i.priority.name == priority.upper()]

    return insights[:limit]

@app.get("/api/insights/{insight_id}/explain")
async def explain_insight(insight_id: str):
    """Get detailed causal explanation for an insight."""
    # Fetch insight and run deeper causal analysis
    pass

@app.post("/api/insights/{insight_id}/acknowledge")
async def acknowledge_insight(insight_id: str, user_id: str):
    """Mark an insight as acknowledged."""
    pass

@app.websocket("/ws/alerts")
async def alert_websocket(websocket: WebSocket):
    """Real-time alert stream via WebSocket."""
    await websocket.accept()

    # Subscribe to alert stream
    async for alert in alert_manager.subscribe():
        await websocket.send_json(alert.dict())

@app.get("/api/metrics/snapshot")
async def get_metrics_snapshot():
    """Get current snapshot of all key metrics across domains."""
    snapshot = {}

    for domain, config in BUSINESS_DOMAINS.items():
        snapshot[domain] = {}
        for metric in config['metrics']:
            snapshot[domain][metric] = {
                'current': get_current_value(domain, metric),
                'previous': get_previous_value(domain, metric),
                'change': calculate_change(domain, metric),
                'trend': get_trend(domain, metric),
            }

    return snapshot

@app.get("/api/causal-graph")
async def get_causal_graph():
    """Get the full causal relationship graph for visualization."""
    return {
        'nodes': [
            {'id': f"{d}.{m}", 'domain': d, 'metric': m}
            for d, cfg in BUSINESS_DOMAINS.items()
            for m in cfg['metrics']
        ],
        'edges': [
            {
                'source': cause,
                'target': effect,
                **attrs
            }
            for (cause, effect), attrs in CROSS_DOMAIN_LINKS.items()
        ]
    }

@app.post("/api/ask")
async def ask_intelligence_agent(query: str):
    """
    Natural language interface to the intelligence agent.

    Examples:
    - "Why is client LTV declining?"
    - "What are the biggest risks this week?"
    - "How is partner performance affecting revenue?"
    """
    # Use LLM to interpret query and route to appropriate analysis
    pass
```

---

## 📁 Final Project Structure

```
enterprise_intelligence/
├── app.py                          # Main FastAPI application
├── index.html                      # Enhanced UI with executive dashboard
│
├── domains/
│   ├── __init__.py
│   ├── domain_registry.py          # Business domain definitions
│   ├── relationship_graph.py       # Cross-domain causal links
│   └── metric_definitions.py       # Metric calculations
│
├── intelligence/
│   ├── __init__.py
│   ├── anomaly_detector.py         # Multi-method anomaly detection
│   ├── causal_reasoner.py          # Causal chain analysis
│   ├── insight_generator.py        # Insight synthesis
│   ├── trend_analyzer.py           # Trend detection
│   └── correlation_finder.py       # Cross-domain correlations
│
├── briefings/
│   ├── __init__.py
│   ├── briefing_generator.py       # Executive briefing generation
│   ├── templates/                  # Briefing templates
│   └── formatters.py               # Output formatters
│
├── alerts/
│   ├── __init__.py
│   ├── alert_manager.py            # Alert rule engine
│   ├── channels/                   # Slack, email, SMS handlers
│   └── rules.yaml                  # Alert rule configurations
│
├── data/
│   ├── connectors/                 # Data source connectors
│   ├── transformers/               # ETL pipelines
│   └── cache/                      # Metric cache
│
└── llm_layer/                      # Existing LLM infrastructure
    ├── smart_extractor.py
    ├── entity_resolver.py
    └── ...
```

---

## 🚀 Implementation Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| **Phase 1** | Weeks 1-3 | Multi-domain data unification, relationship graph |
| **Phase 2** | Weeks 4-7 | Anomaly detection, causal reasoning engine |
| **Phase 3** | Weeks 8-10 | Executive briefing system, alert manager |
| **Phase 4** | Weeks 11-13 | Enhanced UI with causal visualization |
| **Phase 5** | Weeks 14-16 | API layer, integrations, production hardening |

---

## 💡 Key "Wow Factor" Demos

### Demo 1: Causal Chain Explanation
```
User: "Why is client LTV declining?"

Agent: "Client LTV has declined 12% over the past 30 days.

Root cause analysis shows:
1. Partner quality score dropped 18% (90 days ago)
2. This happened because commission rates became 5% below market average
3. Competitor X raised partner payouts 8% on Jan 15th
4. Lower quality partners → lower quality leads → reduced client lifetime value

Recommended actions:
- Review commission structure vs. market
- Implement partner quality tiers
- Consider targeted incentive program

Confidence: 78%"
```

### Demo 2: Morning Briefing
```
📊 EXECUTIVE BRIEFING - Feb 5, 2026

Here's what changed overnight:
• 3 insights requiring attention
• 2 emerging risks
• 1 opportunity identified

EXECUTIVE SUMMARY:
Transaction volume in Palm Jumeirah spiked 34% overnight following the
announcement of the new metro extension. This is driving increased
interest from international buyers (up 28%). However, inventory
constraints in the luxury segment may create pricing pressure.
Recommend focusing today on partner capacity in premium areas.

🔴 REQUIRES ATTENTION:
1. [CRITICAL] Luxury inventory below safety threshold
2. [HIGH] Processing backlog in Dubai Marina office
3. [HIGH] Partner commission disputes up 45%

🟡 EMERGING RISKS:
1. Interest rate hike expected next week - model shows 8% volume impact
2. Competitor launched new agent incentive program

🟢 OPPORTUNITIES:
1. Underserved demand in Dubai Hills - 340 qualified leads, only 12 agents
```

---

## 🎯 Success Metrics

| Metric | Target |
|--------|--------|
| Insight accuracy | >85% validated by business users |
| Causal chain precision | >70% root causes confirmed |
| Alert signal-to-noise | <5% false positive rate |
| Time to insight | <5 minutes from data change |
| Executive adoption | >80% daily briefing open rate |
| Decision impact | Track decisions influenced by agent |

---

This plan transforms your query tool into a proactive intelligence system that doesn't just answer questions—it anticipates them.

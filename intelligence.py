"""
Real-Time Enterprise Intelligence Agent
Hackathon Implementation

Components:
1. MetricTracker - Track key metrics and detect anomalies
2. CausalExplainer - LLM-powered "why" explanations
3. BriefingGenerator - Executive briefing generation
"""

import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import statistics


class InsightType(Enum):
    ATTENTION = "attention"
    RISK = "risk"
    OPPORTUNITY = "opportunity"
    INFO = "info"


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Anomaly:
    """Represents a detected anomaly in the data."""
    anomaly_id: str
    metric_type: str  # price_spike, volume_spike, yield_decline, etc.
    domain: str  # area, property_type, transaction_type
    entity: str  # Palm Jumeirah, Villa, Sales
    current_value: float
    baseline_value: float
    change_percent: float
    severity: Severity
    detected_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            'severity': self.severity.value,
            'detected_at': self.detected_at.isoformat()
        }


@dataclass
class Insight:
    """Represents an actionable insight with causal explanation."""
    insight_id: str
    insight_type: InsightType
    title: str
    summary: str
    explanation: str  # Causal chain
    suggested_action: str
    anomaly: Anomaly
    confidence: float
    affected_metrics: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'insight_id': self.insight_id,
            'insight_type': self.insight_type.value,
            'title': self.title,
            'summary': self.summary,
            'explanation': self.explanation,
            'suggested_action': self.suggested_action,
            'anomaly': self.anomaly.to_dict(),
            'confidence': self.confidence,
            'affected_metrics': self.affected_metrics
        }


@dataclass
class Briefing:
    """Executive briefing with categorized insights."""
    briefing_id: str
    generated_at: datetime
    headline: str
    executive_summary: str
    attention_required: List[Insight]
    risks: List[Insight]
    opportunities: List[Insight]
    metrics_snapshot: Dict[str, Any]

    def to_dict(self) -> dict:
        return {
            'briefing_id': self.briefing_id,
            'generated_at': self.generated_at.isoformat(),
            'headline': self.headline,
            'executive_summary': self.executive_summary,
            'attention_required': [i.to_dict() for i in self.attention_required],
            'risks': [i.to_dict() for i in self.risks],
            'opportunities': [i.to_dict() for i in self.opportunities],
            'metrics_snapshot': self.metrics_snapshot
        }


class MetricTracker:
    """
    Track key real estate metrics and detect significant changes.
    Uses statistical methods to identify anomalies.
    """

    # Thresholds for anomaly detection
    PRICE_CHANGE_THRESHOLD = 0.15  # 15% change
    VOLUME_CHANGE_THRESHOLD = 0.25  # 25% change
    YIELD_CHANGE_THRESHOLD = 0.10  # 10% change

    def __init__(self, db_connection: sqlite3.Connection):
        self.db = db_connection
        self.db.row_factory = sqlite3.Row

    def get_metrics_snapshot(self) -> Dict[str, Any]:
        """Get current state of all key metrics."""
        cursor = self.db.cursor()
        metrics = {}

        # 1. Transaction volume and prices by area
        try:
            cursor.execute("""
                SELECT
                    area_name_en,
                    COUNT(*) as volume,
                    AVG(actual_worth) as avg_price,
                    SUM(actual_worth) as total_value,
                    AVG(meter_sale_price) as avg_psf
                FROM transactions
                WHERE trans_group_en = 'Sales'
                  AND actual_worth > 0
                  AND area_name_en IS NOT NULL
                GROUP BY area_name_en
                ORDER BY volume DESC
                LIMIT 15
            """)
            metrics['sales_by_area'] = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            metrics['sales_by_area'] = []
            metrics['sales_by_area_error'] = str(e)

        # 2. Property type breakdown
        try:
            cursor.execute("""
                SELECT
                    property_type_en,
                    COUNT(*) as count,
                    AVG(actual_worth) as avg_price,
                    SUM(actual_worth) as total_value
                FROM transactions
                WHERE trans_group_en = 'Sales'
                  AND actual_worth > 0
                  AND property_type_en IS NOT NULL
                GROUP BY property_type_en
                ORDER BY count DESC
            """)
            metrics['sales_by_property_type'] = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            metrics['sales_by_property_type'] = []

        # 3. Transaction types distribution
        try:
            cursor.execute("""
                SELECT
                    trans_group_en,
                    COUNT(*) as count,
                    SUM(actual_worth) as total_value
                FROM transactions
                WHERE trans_group_en IS NOT NULL
                GROUP BY trans_group_en
                ORDER BY count DESC
            """)
            metrics['transaction_types'] = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            metrics['transaction_types'] = []

        # 4. Rental metrics by area
        try:
            cursor.execute("""
                SELECT
                    area_name_en,
                    COUNT(*) as contracts,
                    AVG(annual_amount) as avg_rent,
                    SUM(annual_amount) as total_rent_value
                FROM rent_contracts
                WHERE annual_amount > 0
                  AND area_name_en IS NOT NULL
                GROUP BY area_name_en
                ORDER BY contracts DESC
                LIMIT 15
            """)
            metrics['rentals_by_area'] = [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            metrics['rentals_by_area'] = []

        # 5. Calculate rental yields (rent vs sale price) for top areas
        try:
            cursor.execute("""
                SELECT
                    t.area_name_en,
                    AVG(t.actual_worth) as avg_sale_price,
                    (SELECT AVG(r.annual_amount) FROM rent_contracts r WHERE r.area_name_en = t.area_name_en) as avg_rent
                FROM transactions t
                WHERE t.trans_group_en = 'Sales'
                  AND t.actual_worth > 0
                  AND t.area_name_en IS NOT NULL
                GROUP BY t.area_name_en
                HAVING avg_rent IS NOT NULL AND avg_sale_price > 0
                ORDER BY (avg_rent / avg_sale_price) DESC
                LIMIT 10
            """)
            yields = []
            for row in cursor.fetchall():
                if row['avg_sale_price'] and row['avg_rent']:
                    yield_pct = (row['avg_rent'] / row['avg_sale_price']) * 100
                    yields.append({
                        'area': row['area_name_en'],
                        'avg_sale_price': row['avg_sale_price'],
                        'avg_annual_rent': row['avg_rent'],
                        'yield_percent': round(yield_pct, 2)
                    })
            metrics['rental_yields'] = yields
        except Exception as e:
            metrics['rental_yields'] = []

        # 6. Overall market stats
        try:
            cursor.execute("SELECT COUNT(*) as total FROM transactions")
            metrics['total_transactions'] = cursor.fetchone()['total']

            cursor.execute("SELECT COUNT(*) as total FROM rent_contracts")
            metrics['total_rent_contracts'] = cursor.fetchone()['total']

            cursor.execute("SELECT COUNT(DISTINCT area_name_en) as total FROM transactions")
            metrics['unique_areas'] = cursor.fetchone()['total']

            cursor.execute("""
                SELECT SUM(actual_worth) as total
                FROM transactions
                WHERE trans_group_en = 'Sales' AND actual_worth > 0
            """)
            result = cursor.fetchone()
            metrics['total_sales_value'] = result['total'] if result else 0
        except Exception as e:
            pass

        return metrics

    def detect_anomalies(self, metrics: Dict[str, Any]) -> List[Anomaly]:
        """Detect anomalies by comparing against baselines."""
        anomalies = []

        # Calculate baselines from data
        sales_by_area = metrics.get('sales_by_area', [])
        if not sales_by_area:
            return anomalies

        # Calculate overall averages as baseline
        all_volumes = [a['volume'] for a in sales_by_area if a.get('volume')]
        all_prices = [a['avg_price'] for a in sales_by_area if a.get('avg_price')]

        if not all_volumes or not all_prices:
            return anomalies

        avg_volume = statistics.mean(all_volumes)
        std_volume = statistics.stdev(all_volumes) if len(all_volumes) > 1 else avg_volume * 0.3
        avg_price = statistics.mean(all_prices)
        std_price = statistics.stdev(all_prices) if len(all_prices) > 1 else avg_price * 0.3

        # Detect volume anomalies (unusually high activity)
        for area_data in sales_by_area:
            area = area_data.get('area_name_en', 'Unknown')
            volume = area_data.get('volume', 0)
            price = area_data.get('avg_price', 0)

            # Volume spike detection
            if std_volume > 0:
                volume_zscore = (volume - avg_volume) / std_volume
                if volume_zscore > 2.0:  # More than 2 std devs above mean
                    change_pct = ((volume - avg_volume) / avg_volume) * 100
                    anomalies.append(Anomaly(
                        anomaly_id=f"VOL-{area[:10]}-{datetime.now().strftime('%H%M%S')}",
                        metric_type='volume_spike',
                        domain='area',
                        entity=area,
                        current_value=volume,
                        baseline_value=avg_volume,
                        change_percent=change_pct,
                        severity=Severity.HIGH if volume_zscore > 3 else Severity.MEDIUM
                    ))

            # Price anomaly detection
            if std_price > 0 and price > 0:
                price_zscore = (price - avg_price) / std_price
                if abs(price_zscore) > 1.5:
                    change_pct = ((price - avg_price) / avg_price) * 100
                    anomalies.append(Anomaly(
                        anomaly_id=f"PRC-{area[:10]}-{datetime.now().strftime('%H%M%S')}",
                        metric_type='price_spike' if price_zscore > 0 else 'price_drop',
                        domain='area',
                        entity=area,
                        current_value=price,
                        baseline_value=avg_price,
                        change_percent=change_pct,
                        severity=Severity.HIGH if abs(price_zscore) > 2.5 else Severity.MEDIUM
                    ))

        # Detect rental yield anomalies
        yields = metrics.get('rental_yields', [])
        if yields:
            yield_values = [y['yield_percent'] for y in yields]
            avg_yield = statistics.mean(yield_values) if yield_values else 5.0

            for yield_data in yields:
                yield_pct = yield_data.get('yield_percent', 0)
                if yield_pct < avg_yield * 0.7:  # 30% below average yield
                    anomalies.append(Anomaly(
                        anomaly_id=f"YLD-{yield_data['area'][:10]}-{datetime.now().strftime('%H%M%S')}",
                        metric_type='yield_decline',
                        domain='area',
                        entity=yield_data['area'],
                        current_value=yield_pct,
                        baseline_value=avg_yield,
                        change_percent=((yield_pct - avg_yield) / avg_yield) * 100,
                        severity=Severity.MEDIUM
                    ))
                elif yield_pct > avg_yield * 1.3:  # 30% above average yield
                    anomalies.append(Anomaly(
                        anomaly_id=f"YLD-{yield_data['area'][:10]}-{datetime.now().strftime('%H%M%S')}",
                        metric_type='high_yield',
                        domain='area',
                        entity=yield_data['area'],
                        current_value=yield_pct,
                        baseline_value=avg_yield,
                        change_percent=((yield_pct - avg_yield) / avg_yield) * 100,
                        severity=Severity.LOW  # This is opportunity, not risk
                    ))

        # Detect property type concentration
        property_types = metrics.get('sales_by_property_type', [])
        if property_types:
            total_count = sum(p.get('count', 0) for p in property_types)
            for pt in property_types:
                count = pt.get('count', 0)
                share = (count / total_count * 100) if total_count > 0 else 0
                if share > 50:  # One type dominates >50%
                    anomalies.append(Anomaly(
                        anomaly_id=f"CONC-{pt['property_type_en'][:10]}-{datetime.now().strftime('%H%M%S')}",
                        metric_type='market_concentration',
                        domain='property_type',
                        entity=pt['property_type_en'],
                        current_value=share,
                        baseline_value=25,  # Expected ~25% if evenly distributed among 4 types
                        change_percent=share - 25,
                        severity=Severity.LOW
                    ))

        return anomalies


class CausalExplainer:
    """
    Generate causal explanations for anomalies using LLM.
    Uses domain knowledge about Dubai real estate market.
    """

    # Known causal patterns in Dubai real estate
    CAUSAL_PATTERNS = {
        'price_spike': [
            "Supply constraints due to limited new inventory",
            "Infrastructure development (metro extension, new roads)",
            "Regulatory changes (visa reforms, ownership laws)",
            "Seasonal demand from international investors (Q4)",
            "Developer payment plan promotions ending",
            "Nearby landmark/mall opening"
        ],
        'price_drop': [
            "New project completions increasing supply",
            "Interest rate increases affecting affordability",
            "Economic uncertainty or market correction",
            "Oversupply in specific property segments",
            "Shift in buyer preferences to other areas"
        ],
        'volume_spike': [
            "New project launches with attractive pricing",
            "End of developer moratorium periods",
            "Seasonal investor activity (Q4, Ramadan)",
            "Government incentive programs",
            "Expo/major event driving interest"
        ],
        'yield_decline': [
            "Sale prices rising faster than rental rates",
            "Oversupply in rental market",
            "Tenant preference shifting to newer areas",
            "Economic factors affecting rental demand"
        ],
        'high_yield': [
            "Undervalued area with strong rental demand",
            "Corporate housing requirements",
            "Limited supply keeping rents high",
            "Area becoming popular for short-term rentals"
        ],
        'market_concentration': [
            "Market segment dominance in the area",
            "Developer focus on specific property types",
            "Buyer preference patterns",
            "Zoning and development regulations"
        ]
    }

    # Recommended actions by anomaly type
    ACTIONS = {
        'price_spike': "Monitor inventory levels, alert sellers to capitalize on demand, review pricing strategy",
        'price_drop': "Pause aggressive listings, focus on buyer acquisition, highlight value proposition",
        'volume_spike': "Increase agent capacity in the area, expedite transaction processing, prepare marketing materials",
        'yield_decline': "Shift marketing focus to capital appreciation story, target different buyer segments",
        'high_yield': "Target income-focused investors, highlight rental ROI in marketing",
        'market_concentration': "Diversify portfolio focus, identify emerging segments"
    }

    def __init__(self, llm_model, db_connection: sqlite3.Connection):
        self.llm = llm_model
        self.db = db_connection
        self.db.row_factory = sqlite3.Row

    def explain(self, anomaly: Anomaly) -> Insight:
        """Generate causal explanation for an anomaly."""

        # Get additional context from database
        context = self._get_context(anomaly)

        # Get relevant causal patterns
        patterns = self.CAUSAL_PATTERNS.get(anomaly.metric_type, ["Market dynamics"])

        # Build explanation using LLM if available
        if self.llm:
            explanation = self._generate_llm_explanation(anomaly, context, patterns)
        else:
            explanation = self._generate_fallback_explanation(anomaly, patterns)

        # Determine insight type
        if anomaly.metric_type in ['price_spike', 'volume_spike', 'high_yield']:
            insight_type = InsightType.OPPORTUNITY
        elif anomaly.metric_type in ['price_drop', 'yield_decline']:
            insight_type = InsightType.RISK
        elif anomaly.severity in [Severity.CRITICAL, Severity.HIGH]:
            insight_type = InsightType.ATTENTION
        else:
            insight_type = InsightType.INFO

        # Override to ATTENTION if severity is critical
        if anomaly.severity == Severity.CRITICAL:
            insight_type = InsightType.ATTENTION

        # Generate title — concise and scannable
        arrow = "↑" if anomaly.change_percent > 0 else "↓"
        metric_labels = {
            'price_spike': 'Prices',
            'price_drop': 'Prices',
            'volume_spike': 'Transaction Volume',
            'yield_decline': 'Rental Yield',
            'high_yield': 'Rental Yield',
            'market_concentration': 'Market Concentration',
        }
        metric_label = metric_labels.get(anomaly.metric_type, anomaly.metric_type.replace('_', ' ').title())
        title = f"{anomaly.entity} — {metric_label} {arrow}{abs(anomaly.change_percent):.0f}%"

        # Generate summary
        summary = title

        return Insight(
            insight_id=f"INS-{anomaly.anomaly_id}",
            insight_type=insight_type,
            title=title,
            summary=summary,
            explanation=explanation,
            suggested_action=self.ACTIONS.get(anomaly.metric_type, "Monitor closely"),
            anomaly=anomaly,
            confidence=0.75 if self.llm else 0.60,
            affected_metrics=[anomaly.metric_type, anomaly.domain]
        )

    def _get_context(self, anomaly: Anomaly) -> Dict[str, Any]:
        """Get relevant context data for the anomaly."""
        context = {}
        cursor = self.db.cursor()

        try:
            if anomaly.domain == 'area':
                # Get property type breakdown for this area
                cursor.execute("""
                    SELECT property_type_en, COUNT(*) as cnt, AVG(actual_worth) as avg_price
                    FROM transactions
                    WHERE area_name_en = ? AND trans_group_en = 'Sales'
                    GROUP BY property_type_en
                    ORDER BY cnt DESC
                    LIMIT 5
                """, (anomaly.entity,))
                context['property_breakdown'] = [dict(row) for row in cursor.fetchall()]

                # Get transaction type breakdown
                cursor.execute("""
                    SELECT trans_group_en, COUNT(*) as cnt
                    FROM transactions
                    WHERE area_name_en = ?
                    GROUP BY trans_group_en
                    ORDER BY cnt DESC
                """, (anomaly.entity,))
                context['transaction_types'] = [dict(row) for row in cursor.fetchall()]

                # Get rental data for the area
                cursor.execute("""
                    SELECT COUNT(*) as contracts, AVG(annual_amount) as avg_rent
                    FROM rent_contracts
                    WHERE area_name_en = ?
                """, (anomaly.entity,))
                row = cursor.fetchone()
                if row:
                    context['rental_data'] = dict(row)
        except Exception as e:
            context['error'] = str(e)

        return context

    def _generate_llm_explanation(self, anomaly: Anomaly, context: Dict, patterns: List[str]) -> str:
        """Generate explanation using LLM."""

        prompt = f"""You are a Dubai real estate market analyst. Generate a brief causal explanation for this market observation.

OBSERVATION:
- Metric: {anomaly.metric_type.replace('_', ' ')}
- Location/Entity: {anomaly.entity}
- Current Value: {anomaly.current_value:,.0f}
- Baseline Value: {anomaly.baseline_value:,.0f}
- Change: {anomaly.change_percent:+.1f}%

CONTEXT DATA:
{json.dumps(context, indent=2, default=str)}

KNOWN FACTORS that typically cause {anomaly.metric_type.replace('_', ' ')}:
{chr(10).join(f'- {p}' for p in patterns)}

Generate a 2-3 sentence causal explanation that:
1. States what is happening with specific numbers
2. Explains the most likely cause based on the data
3. Notes what this suggests for the near future

Format: "[Area/Entity] is experiencing [change] because [cause]. This is likely driven by [factor]. [Future implication]."

Be specific and data-driven. Use numbers where available."""

        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return self._generate_fallback_explanation(anomaly, patterns)

    def _generate_fallback_explanation(self, anomaly: Anomaly, patterns: List[str]) -> str:
        """Generate explanation without LLM."""
        direction = "up" if anomaly.change_percent > 0 else "down"
        cause = patterns[0] if patterns else "market dynamics"
        # Pick a second cause for variety if available
        cause2 = patterns[1] if len(patterns) > 1 else None

        # Format values nicely
        def _fmt(v):
            if abs(v) >= 1e6: return f"AED {v/1e6:.1f}M"
            if abs(v) >= 1e3: return f"AED {v/1e3:.0f}K"
            if abs(v) >= 100: return f"{v:,.0f}"
            return f"{v:.1f}%"  # likely a percentage/yield

        current_fmt = _fmt(anomaly.current_value)
        baseline_fmt = _fmt(anomaly.baseline_value)

        explanation = f"{direction.title()} {abs(anomaly.change_percent):.0f}% vs market average ({current_fmt} vs {baseline_fmt}). Likely driven by {cause.lower()}."
        if cause2:
            explanation += f" {cause2}."

        return explanation


class BriefingGenerator:
    """Generate executive briefings from metrics and insights."""

    def __init__(self, llm_model, metric_tracker: MetricTracker, causal_explainer: CausalExplainer):
        self.llm = llm_model
        self.tracker = metric_tracker
        self.explainer = causal_explainer

    def generate_briefing(self) -> Briefing:
        """Generate complete intelligence briefing."""

        # 1. Get current metrics
        metrics = self.tracker.get_metrics_snapshot()

        # 2. Detect anomalies
        anomalies = self.tracker.detect_anomalies(metrics)

        # 3. Generate insights for each anomaly
        insights = []
        for anomaly in anomalies[:10]:  # Limit to top 10
            insight = self.explainer.explain(anomaly)
            insights.append(insight)

        # 4. Categorize insights
        attention = [i for i in insights if i.insight_type == InsightType.ATTENTION]
        risks = [i for i in insights if i.insight_type == InsightType.RISK]
        opportunities = [i for i in insights if i.insight_type == InsightType.OPPORTUNITY]

        # Add some insights to attention if critical
        for i in insights:
            if i.anomaly.severity in [Severity.CRITICAL, Severity.HIGH] and i not in attention:
                attention.append(i)

        # 5. Generate headline
        headline = self._generate_headline(attention, risks, opportunities)

        # 6. Generate executive summary
        exec_summary = self._generate_executive_summary(insights, metrics)

        return Briefing(
            briefing_id=f"BRIEF-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            generated_at=datetime.now(),
            headline=headline,
            executive_summary=exec_summary,
            attention_required=attention[:5],
            risks=risks[:3],
            opportunities=opportunities[:3],
            metrics_snapshot=metrics
        )

    def _generate_headline(self, attention: List, risks: List, opportunities: List) -> str:
        """Generate the briefing headline."""
        parts = []

        if attention:
            parts.append(f"{len(attention)} insight{'s' if len(attention) != 1 else ''} requiring attention")
        if risks:
            parts.append(f"{len(risks)} emerging risk{'s' if len(risks) != 1 else ''}")
        if opportunities:
            parts.append(f"{len(opportunities)} opportunit{'ies' if len(opportunities) != 1 else 'y'} identified")

        if not parts:
            return "All systems nominal - market conditions stable"

        return "Here's what changed: " + ", ".join(parts)

    def _generate_executive_summary(self, insights: List[Insight], metrics: Dict) -> str:
        """Generate executive summary using LLM or fallback."""

        if not insights:
            return "No significant market changes detected. All metrics within normal ranges."

        if self.llm:
            return self._generate_llm_summary(insights, metrics)
        else:
            return self._generate_fallback_summary(insights, metrics)

    def _generate_llm_summary(self, insights: List[Insight], metrics: Dict) -> str:
        """Generate summary using LLM."""

        insight_texts = "\n".join([
            f"- [{i.insight_type.value.upper()}] {i.title}: {i.explanation[:200]}"
            for i in insights[:5]
        ])

        # Key metrics
        total_trans = metrics.get('total_transactions', 0)
        total_value = metrics.get('total_sales_value', 0)

        prompt = f"""Generate a concise 3-4 sentence executive briefing for a Dubai real estate intelligence report.

KEY METRICS:
- Total Transactions: {total_trans:,}
- Total Sales Value: AED {total_value:,.0f}
- Areas Tracked: {metrics.get('unique_areas', 0)}

TOP INSIGHTS:
{insight_texts}

Requirements:
- Lead with the most impactful finding
- Use specific numbers and percentages
- Be direct and actionable
- End with recommended focus area

Write in confident executive tone. No bullet points."""

        try:
            response = self.llm.generate_content(prompt)
            return response.text.strip()
        except Exception:
            return self._generate_fallback_summary(insights, metrics)

    def _generate_fallback_summary(self, insights: List[Insight], metrics: Dict) -> str:
        """Generate summary without LLM."""

        if not insights:
            return "Market conditions are stable with no significant anomalies detected."

        top_insight = insights[0]
        total_value = metrics.get('total_sales_value', 0)

        return (
            f"Key finding: {top_insight.title} - {top_insight.summary}. "
            f"The market shows AED {total_value/1e9:.1f}B in total sales value across "
            f"{metrics.get('unique_areas', 0)} areas. "
            f"Recommended action: {top_insight.suggested_action}."
        )


# Convenience function to create all components
def create_intelligence_system(db_connection: sqlite3.Connection, llm_model=None):
    """Factory function to create the intelligence system."""
    tracker = MetricTracker(db_connection)
    explainer = CausalExplainer(llm_model, db_connection)
    generator = BriefingGenerator(llm_model, tracker, explainer)

    return {
        'tracker': tracker,
        'explainer': explainer,
        'generator': generator
    }

"""Insider ownership event extraction."""
from __future__ import annotations

import logging
from typing import Dict, List

from edgar import Filing

logger = logging.getLogger(__name__)


def extract_insider_events(filing: Filing) -> List[Dict]:
    logger.debug(
        "extract_insider_events entry", extra={"form": filing.form, "filing_date": filing.filing_date, "text_url": getattr(filing, "text_url", "")}
    )

    events: List[Dict] = []
    try:
        obj = filing.obj()
    except Exception:
        obj = None
        logger.debug("filing.obj() failed", exc_info=True)

    if obj is not None:
        parse_status = "edgar_obj_ok"
        transactions = getattr(obj, "transactions", []) or []
        for txn in transactions:
            events.append(
                {
                    "transaction_date": getattr(txn, "transaction_date", None),
                    "transaction_code": getattr(txn, "transaction_code", None),
                    "derivative_flag": getattr(txn, "is_derivative", None),
                    "shares": getattr(txn, "shares", None),
                    "price_per_share": getattr(txn, "price", None),
                    "value": getattr(txn, "value", None),
                    "ownership_type": getattr(txn, "ownership_type", None),
                    "parse_status": parse_status,
                }
            )

    logger.debug(
        "extract_insider_events exit", extra={"form": filing.form, "filing_date": filing.filing_date, "events": len(events)}
    )
    return events


def aggregate_insider_events(events: List[Dict]) -> Dict:
    buy_count = 0
    sell_count = 0
    net_shares = 0

    for event in events:
        code = (event.get("transaction_code") or "").upper()
        shares = event.get("shares") or 0
        if code.startswith("P"):
            buy_count += 1
            try:
                net_shares += float(shares)
            except Exception:
                pass
        elif code.startswith("S"):
            sell_count += 1
            try:
                net_shares -= float(shares)
            except Exception:
                pass

    return {
        "buy_count": buy_count,
        "sell_count": sell_count,
        "net_shares_bought": net_shares,
        "cluster_flag": buy_count + sell_count > 1,
    }

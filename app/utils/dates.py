from datetime import date, datetime
from typing import Optional


def formatar_data(dt: datetime, formato: str = "%Y-%m-%d") -> str:
    return dt.strftime(formato)


def parse_data(valor: str, formato: str = "%Y-%m-%d") -> Optional[date]:
    try:
        return datetime.strptime(valor, formato).date()
    except (ValueError, TypeError):
        return None


def hoje() -> date:
    return date.today()

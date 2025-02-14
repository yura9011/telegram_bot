import datetime
import logging
from typing import Optional

logger = logging.getLogger(__name__)

async def parse_datetime(date_str: Optional[str], time_str: Optional[str]) -> Optional[str]:
    """
    Parses date and time strings into ISO format datetime string.
    Handles cases where date or time are missing, defaulting to current date/time.
    """
    if not date_str:
        date_part = datetime.date.today().isoformat() # Default to today's date
    else:
        try:
            datetime.date.fromisoformat(date_str) # Validate date format
            date_part = date_str
        except ValueError:
            return None # Invalid date format

    if not time_str:
        time_part = datetime.datetime.now().strftime("%H:%M") # Default to current time (HH:MM)
    else:
        try:
            datetime.datetime.strptime(time_str, "%H:%M").time() # Validate time format
            time_part = time_str
        except ValueError:
            return None # Invalid time format

    try:
        # Combine date and time parts into ISO format datetime string
        combined_datetime = f"{date_part}T{time_part}:00" # Seconds are set to 00
        datetime.datetime.fromisoformat(combined_datetime) # Validate combined format
        return combined_datetime

    except ValueError:
        return None # Invalid combined datetime format
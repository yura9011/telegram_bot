import re

def escape_markdown_v2(text: str) -> str:
    """
    Escapes characters reserved in Telegram MarkdownV2 format.
    
    Args:
        text: The text to escape
    
    Returns:
        Escaped text safe for use with MarkdownV2
    """
    # Characters that need to be escaped in MarkdownV2
    reserved_characters = r'[_*\[\]()~`>#+\-=|{}.!]'
    return re.sub(f'({reserved_characters})', r'\\\1', text)
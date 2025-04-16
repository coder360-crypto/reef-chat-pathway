import pymupdf  # import package PyMuPDF
import textwrap
import logging
import json

logger = logging.getLogger(__name__)


def add_highlights(highlight_data, filename) -> str:
    """
    Add highlights and suggestions to a PDF file and create a summary page.
    
    Args:
        highlight_data (list): List of dictionaries containing content and suggested changes
        filename (str): Path to the PDF file
        
    Returns:
        str: Path to the highlighted PDF file
    """
    # Define colors for different elements
    title_color = (0, 0.2, 0.4)  # Darker blue for title
    section_color = (0.1, 0.1, 0.1)  # Almost black for section numbers
    original_color = (0.3, 0.3, 0.3)  # Darker gray for original text
    suggestion_color = (0, 0.4, 0.2)  # Dark green for suggestions
    highlight_color = (1, 0.8, 0.8)
    strikeout_color = (0.3, 0.3, 0.3)
    page_ref_color = (0.4, 0.4, 0.4)  # Medium gray for page references

    # Log highlight data and save to file
    logger.info(f"Highlight data: {highlight_data}")
    with open("highlight_data.json", "w") as f:
        json.dump(highlight_data, f)
    # return "highlighted_" + filename
    # Open document
    doc = pymupdf.open(filename)

    # Process each highlight
    for highlight in highlight_data:
        content = highlight["content"]
        highlight["page"] = 0
        for i, page in enumerate(doc):
            rects = page.search_for(content)
            if rects:
                highlight["page"] = i
                # Add highlight annotation
                annot = page.add_highlight_annot(rects)
                annot.set_colors(stroke=highlight_color)
                annot.set_info(
                    content=highlight["suggested_content"]
                )  # Add suggestion as note
                annot.update()

                # Add strikeout annotation
                strikeout = page.add_strikeout_annot(rects)
                strikeout.set_colors(stroke=strikeout_color)
                strikeout.update()

    # Get page dimensions
    first_page = doc[0]
    page_width = first_page.rect.width
    page_height = first_page.rect.height

    # Create new page for suggestions
    suggestions_page = doc.new_page(width=page_width, height=page_height)

    # Define layout parameters
    margin_left = 72  # 1 inch margin
    margin_right = 72
    margin_top = 72
    line_spacing = 18
    text_width = page_width - margin_left - margin_right

    # Add title to suggestions page
    y_position = margin_top
    title = "Suggested Changes"
    suggestions_page.insert_text(
        (margin_left, y_position),
        title,
        fontsize=18,
        fontname="Helvetica-Bold",
        color=title_color,
    )
    y_position += 36

    def wrap_text(text, width=60):
        """
        Wrap text to a specified width.
        
        Args:
            text (str): Text to wrap
            width (int): Maximum width in characters
            
        Returns:
            str: Wrapped text
        """
        return textwrap.fill(text, width=width)

    # Track current page in suggestions section
    suggestions_page_index = len(doc) - 1

    # Add each suggestion to the page
    for i, highlight in enumerate(highlight_data, start=1):
        # Check if we need a new page
        if y_position > page_height - 100:
            suggestions_page = doc.new_page(width=page_width, height=page_height)
            suggestions_page_index += 1
            y_position = margin_top

        # Update text colors
        suggestions_page.insert_text(
            (margin_left, y_position),
            f"{i}. ",
            fontsize=12,  # Slightly larger
            fontname="Helvetica-Bold",
            color=section_color,
        )

        # Original text with updated color
        orig_text = wrap_text(highlight["content"])

        # Add a clickable link to the original content
        orig_rects = doc[highlight["page"]].search_for(highlight["content"])

        # Insert text
        text_bbox = suggestions_page.insert_text(
            (margin_left + 20, y_position),
            "Original: " + orig_text,
            fontsize=11,
            fontname="Helvetica",
            color=original_color,
        )

        # Insert link to original page and location
        if orig_rects:
            # Create a link annotation
            link_rect = pymupdf.Rect(
                margin_left + 20,
                y_position,
                page_width - margin_right,
                y_position + line_spacing * (orig_text.count("\n") + 1),
            )
            link = suggestions_page.insert_link(
                {
                    "kind": pymupdf.LINK_GOTO,
                    "page": highlight["page"],
                    "to": pymupdf.Point(orig_rects[0].x0, orig_rects[0].y0),
                    "from": link_rect,
                }
            )

        y_position += line_spacing * (orig_text.count("\n") + 1)

        # Suggested text with updated color
        sugg_text = wrap_text(highlight["suggested_content"])
        suggestions_page.insert_text(
            (margin_left + 20, y_position),
            "Suggestion: " + sugg_text,
            fontsize=11,
            fontname="Helvetica-Bold",
            color=suggestion_color,
        )
        y_position += line_spacing * (sugg_text.count("\n") + 1)

        # Page reference with updated color
        suggestions_page.insert_text(
            (page_width - margin_right - 50, y_position),
            f"(Page {highlight['page'] + 1})",
            fontsize=10,
            fontname="Helvetica",
            color=page_ref_color,
        )

        # Add some extra spacing between suggestions
        y_position += line_spacing * 1.5

    # Save the document
    doc.save("files/" + "highlighted_" + filename)
    return "files/" + "highlighted_" + filename


if __name__ == "__main__":
    with open("highlight_data.json", "r") as f:
        highlight_data = json.load(f)
    add_highlights(highlight_data, "nda.pdf")

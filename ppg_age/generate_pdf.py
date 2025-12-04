
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

def parse_markdown_table(lines):
    """Parses a markdown table into a list of lists."""
    table_data = []
    for line in lines:
        if line.strip().startswith('|'):
            # Remove leading/trailing pipes and split
            row = [cell.strip() for cell in line.strip().strip('|').split('|')]
            table_data.append(row)
    
    # Remove the separator line (second line usually)
    if len(table_data) > 1:
        # Check if the second row is just dashes
        if set(table_data[1][0]).issubset({'-', ':', ' '}):
            table_data.pop(1)
            
    return table_data

def generate_pdf(md_file, pdf_file):
    doc = SimpleDocTemplate(pdf_file, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    with open(md_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if not line:
            i += 1
            continue
            
        # Headers
        if line.startswith('# '):
            story.append(Paragraph(line[2:], styles['Title']))
            story.append(Spacer(1, 12))
        elif line.startswith('## '):
            story.append(Paragraph(line[3:], styles['Heading2']))
            story.append(Spacer(1, 10))
        elif line.startswith('### '):
            story.append(Paragraph(line[4:], styles['Heading3']))
            story.append(Spacer(1, 8))
            
        # Images
        elif line.startswith('![') and '](' in line:
            # Extract path
            start = line.find('](') + 2
            end = line.find(')', start)
            img_path = line[start:end]
            if os.path.exists(img_path):
                # Scale image to fit page width
                img = Image(img_path)
                img_width = 6 * inch # Approx page width
                aspect = img.imageHeight / img.imageWidth
                img.drawHeight = img_width * aspect
                img.drawWidth = img_width
                story.append(img)
                story.append(Spacer(1, 12))
            else:
                story.append(Paragraph(f"[Image not found: {img_path}]", styles['BodyText']))
                
        # Tables
        elif line.startswith('|'):
            # Collect all table lines
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
            i -= 1 # Backtrack one since the loop overshot
            
            data = parse_markdown_table(table_lines)
            if data:
                t = Table(data)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ]))
                story.append(t)
                story.append(Spacer(1, 12))
                
        # Lists
        elif line.startswith('- ') or line.startswith('* ') or (line[0].isdigit() and line[1:3] == '. '):
            # Simple list handling
            text = line.split(' ', 1)[1]
            # Check for bolding **text**
            import re
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
            story.append(Paragraph(f"• {text}", styles['BodyText']))
            
        # Normal Text
        else:
            # Check for bolding
            import re
            text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            story.append(Paragraph(text, styles['BodyText']))
            story.append(Spacer(1, 6))
            
        i += 1

    doc.build(story)
    print(f"PDF generated: {pdf_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Convert Markdown to PDF.')
    parser.add_argument('input', help='Path to input Markdown file')
    parser.add_argument('output', help='Path to output PDF file')
    args = parser.parse_args()
    
    generate_pdf(args.input, args.output)

import requests
from bs4 import BeautifulSoup
import csv
import os  # To check if file exists

# URL of the arXiv recent submissions in Computer Vision (cs.CV)
url = "https://arxiv.org/list/cs.CV/recent"

# Fetch the main page
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Find the papers listed on the page
papers = soup.find_all('div', class_='meta')[0:5]  # Get first 5 papers

# List to hold metadata dictionaries
metadata_list = []

for idx, paper in enumerate(papers, 1):
    try:
        # Extract metadata
        title = paper.find('div', class_='list-title').text.strip().replace('Title:\n', '').strip()
        authors = paper.find('div', class_='list-authors').text.strip().replace('Authors:', '').strip()

        # Extract the arXiv ID from the sibling <dt> tag
        arxiv_id = None
        dt_tag = paper.find_previous('dt')  # Find the previous <dt> tag
        if dt_tag:
            link = dt_tag.find('a', href=True)  # Look for <a> tags with href
            if link and '/abs/' in link['href']:
                arxiv_id = link['href'].split('/')[-1]  # Extract the ID from the href

        publication_date = "No date available."
        if arxiv_id and len(arxiv_id) >= 4:
            year = "20" + arxiv_id[:2]  # First two digits are the year, '24' -> 2024
            month = arxiv_id[2:4]       # Next two digits are the month, '09' -> September
            publication_date = f"{year}-{month}"
        # Fetch the abstract from the individual paper page
        abstract = "No abstract available."
        if arxiv_id:
            abstract_url = f"https://arxiv.org/abs/{arxiv_id}"
            abstract_response = requests.get(abstract_url)
            abstract_soup = BeautifulSoup(abstract_response.content, 'html.parser')

            # Navigate to the abstract section
            abstract_block = abstract_soup.find('blockquote', class_='abstract')
            if abstract_block:
                # Remove 'Abstract: ' from the start of the abstract text
                abstract = abstract_block.text.strip().replace('Abstract:', '').replace('\n', ' ').strip()

        # Construct the PDF link correctly
        pdf_link = f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else "No PDF link available."

        # Print the metadata
        print(f"Paper {idx}:")
        print(f"Title: {title}")
        print(f"Authors: {authors}")
        print(f"Abstract: {abstract}")
        print(f"PDF Link: {pdf_link}")
        print("-" * 80)

        # Create a dictionary for the current paper's metadata
        paper_metadata = {
            'arxiv_id': arxiv_id,
            'Author': authors,
            'Title': title,
            'Abstract': abstract,
            'publication_date': publication_date
        }
        metadata_list.append(paper_metadata)  # Add the dictionary to the list

        # Check if the PDF is already stored
        pdf_filename = f"paper_{arxiv_id}.pdf"
        if arxiv_id and not os.path.exists(pdf_filename):
            pdf_response = requests.get(pdf_link)
            with open(pdf_filename, 'wb') as pdf_file:  # Save using arXiv ID
                pdf_file.write(pdf_response.content)
            print(f"PDF for Paper {idx} downloaded.\n")
        else:
            print(f"PDF for Paper {idx} already exists, skipping download.\n")

    except Exception as e:
        print(f"An error occurred while processing paper {idx}: {e}")
        print("-" * 80)

# Save the metadata to a CSV file after processing all papers
with open('arxiv_metadata.csv', 'w', newline='', encoding='utf-8') as csvfile:
    # Write the header
    writer = csv.DictWriter(csvfile, fieldnames=['arxiv_id','Author', 'Title', 'Abstract','publication_date'], quoting=csv.QUOTE_ALL)
    
    # Write the header
    writer.writeheader()
    
    # Write each paper's metadata as a CSV row
    for data in metadata_list:
        writer.writerow(data)

print("Metadata saved to arxiv_metadata.csv.")
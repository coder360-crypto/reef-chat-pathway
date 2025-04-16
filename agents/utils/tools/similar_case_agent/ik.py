# Import system modules
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import external dependencies
import requests
import logging
from bs4 import BeautifulSoup

# Import local config
from config import AgentsConfig as Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

config = Config()

class IKApi:
    """
    Indian Kanoon API wrapper for fetching and processing legal documents.
    
    Args:
        token (str): Authentication token for the API
        base_url (str): Base URL for the API endpoints
    """
    def __init__(self, token, base_url="https://api.indiankanoon.org"):
        # Initialize API headers and base URL
        self.headers = {
            'Authorization': f'Token {token}',
            'Accept': 'application/json'
        }
        self.base_url = base_url

    def call_api(self, url, max_retries=3, retry_delay=3):
        """Make API call and return JSON response with retry logic
        
        Args:
            url (str): API endpoint
            max_retries (int): Maximum number of retry attempts
            retry_delay (int): Delay between retries in seconds
            
        Returns:
            dict: JSON response from API
        """
        from time import sleep
        
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.base_url}{url}", 
                    headers=self.headers,
                    timeout=10  # Add timeout
                )
                response.raise_for_status()  # Raise exception for bad status codes
                return response.json()
            except (requests.exceptions.RequestException, ConnectionError) as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"Failed to call API after {max_retries} attempts: {str(e)}")
                    raise
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                sleep(retry_delay)

    def fetch_doc(self, docid):
        """Fetch full document by ID"""
        url = f'/doc/{docid}/'
        return self.call_api(url)

    def get_doc_ids(self, query, doctypes="judgments"):
        """Search documents with query"""
        url = f'/search/?formInput={query}+doctypes:{doctypes}'
        data = self.call_api(url)["docs"]
        return [doc["tid"] for doc in data]

    def fetch_doc(self, docid):
        """Fetch document fragment matching query"""
        url = f'/doc/{docid}/'
        return self.call_api(url)
    
    def scrape_document_details(self, query, doctypes="judgments", max_documents=3):
        """Fetch and process multiple documents
        
        Args:
            query (str): Search query
            
        Returns:
            dict: Contains status, data list, and any error messages
        """
        scraped_data_list = []
        docids = self.get_doc_ids(query, doctypes)[:max_documents]
        try:
            for docid in docids:
                doc_data = self.fetch_doc(docid)
                if doctypes == "judgments":
                    processed_data = self._process_judgments(doc_data)
                else:
                    processed_data = self._process_laws(doc_data)
                if processed_data:
                    scraped_data_list.append(processed_data)
                    
            return {
                "status": "success",
                "data": scraped_data_list,
                "error": None
            }
        except Exception as e:
            return {
                "status": "error",
                "data": scraped_data_list,
                "error": str(e)
            }

    def _process_judgments(self, doc_data):
        """Process raw document data into structured format
        
        Args:
            doc_data (dict): Raw API response for a document
            
        Returns:
            dict: Processed document data with title, background, conclusion etc.
        """
        try:
            soup = BeautifulSoup(doc_data['doc'], 'html.parser')
            
            # Extract title
            doc_title = soup.find('h2', class_='doc_title').text.strip() if soup.find('h2', class_='doc_title') else ""
            
            # Updated to use data-structure instead of title
            background_paras = soup.find_all(['p', 'blockquote'], attrs={'data-structure': lambda x: x in ['Issue', 'Fact']})
            conclusion_paras = soup.find_all(['p'], attrs={'data-structure': 'Conclusion'})
            
            background_text = "\n".join(p.text.strip() for p in background_paras) if background_paras else ""
            conclusion_text = "\n".join(p.text.strip() for p in conclusion_paras) if conclusion_paras else ""
            
            return {
                "doc_title": doc_title,
                "background": background_text,
                "conclusion": conclusion_text,
                "url": f"https://indiankanoon.org/doc/{doc_data['tid']}/"
            }
        except Exception as e:
            logger.error(f"Error processing document {doc_data.get('tid')}: {str(e)}")
            return None
        
    def _process_laws(self, doc_data):
        """Process raw document data into structured format
        
        Args:
            doc_data (dict): Raw API response for a document
            
        Returns:
            dict: Processed document data with title, content and url
        """
        try:
            soup = BeautifulSoup(doc_data['doc'], 'html.parser')
            
            # Extract title
            doc_title = soup.find('h2', class_='doc_title').text.strip() if soup.find('h2', class_='doc_title') else ""
            
            # Extract content from akn-section elements
            akn_sections = soup.find_all(class_='akn-section')
            content = "\n".join(section.text.strip() for section in akn_sections) if akn_sections else ""

            return {
                "doc_title": doc_title,
                "content": content,
                "url": f"https://indiankanoon.org/doc/{doc_data['tid']}/"
            }
        except Exception as e:
            logger.error(f"Error processing law document {doc_data.get('tid')}: {str(e)}")
            return None

if __name__ == "__main__":
    import json
    ik_api = IKApi(token=config.IK_API_KEY)
    

    data = ik_api.scrape_document_details("accidental death report in domestic disputes", max_documents=5, doctypes="laws")
    with open('temp3.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
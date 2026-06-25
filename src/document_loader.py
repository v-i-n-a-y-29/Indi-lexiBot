import os
import re
import json
import hashlib
from pathlib import Path
from langchain_community.document_loaders import PDFPlumberLoader
from cleaner import clean_text

DATA_DIR = Path("data")
PROCESSED_DIR = Path("processed")
OUTPUT_FILE = PROCESSED_DIR / "clean_documents.jsonl"
REPORT_FILE = PROCESSED_DIR / "metadata_report.json"

def compute_file_hash(file_path: Path) -> str:
    """Computes MD5 hash of a file to detect exact duplicates."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read(65536)
        while len(buf) > 0:
            hasher.update(buf)
            buf = f.read(65536)
    return hasher.hexdigest()

def extract_metadata_from_filename(filename: str) -> dict:
    """Extracts case name and date from filenames if they match known patterns."""
    meta = {"case_name": None, "judgment_date": None}
    
    # Remove extension
    name_no_ext = os.path.splitext(filename)[0]
    # Remove trailing duplicate markers like (1), (2)
    name_no_ext = re.sub(r'\s*\(\d+\)$', '', name_no_ext)
    
    # Pattern: Case_Name_on_Date
    match_on = re.search(r'(.+?)_on_(.+)', name_no_ext)
    if match_on:
        meta["case_name"] = match_on.group(1).replace('_', ' ').strip()
        meta["judgment_date"] = match_on.group(2).replace('_', ' ').strip()
        return meta
        
    # Pattern: Section_X_in_Act_Y
    match_in = re.search(r'(.+?)_in_(.+)', name_no_ext)
    if match_in:
        meta["case_name"] = name_no_ext.replace('_', ' ').strip()
        # Extract 4-digit year as a fallback date if present
        year_match = re.search(r'(19|20)\d{2}', match_in.group(2))
        if year_match:
            meta["judgment_date"] = year_match.group(0)
            
    return meta

def run_pipeline():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    seen_hashes = set()
    report = {
        "total_files_found": 0,
        "files_processed": 0,
        "duplicates_skipped": 0,
        "failed_files": [],
        "total_pages_extracted": 0
    }
    
    pdf_files = list(DATA_DIR.rglob("*.pdf")) + list(DATA_DIR.rglob("*.PDF"))
    report["total_files_found"] = len(pdf_files)
    
    print(f"Found {len(pdf_files)} PDF files.")
    
    # Open JSONL file for appending or writing
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for pdf_path in pdf_files:
            try:
                file_hash = compute_file_hash(pdf_path)
                if file_hash in seen_hashes:
                    report["duplicates_skipped"] += 1
                    continue
                seen_hashes.add(file_hash)
                
                domain = pdf_path.parent.name
                filename = pdf_path.name
                file_meta = extract_metadata_from_filename(filename)
                
                loader = PDFPlumberLoader(str(pdf_path))
                docs = loader.load()
                
                if not docs:
                    report["failed_files"].append({
                        "path": str(pdf_path),
                        "reason": "No text extracted (possible scanned PDF)"
                    })
                    continue
                    
                has_text = False
                for i, doc in enumerate(docs):
                    raw_text = doc.page_content
                    cleaned_text = clean_text(raw_text)
                    
                    if not cleaned_text.strip():
                        continue
                        
                    has_text = True
                    page_num = i + 1
                    
                    record = {
                        "doc_id": f"{file_hash}_p{page_num}",
                        "text": cleaned_text,
                        "metadata": {
                            "source_path": str(pdf_path),
                            "file_name": filename,
                            "domain": domain,
                            "page_number": page_num,
                            "case_name": file_meta["case_name"],
                            "judgment_date": file_meta["judgment_date"]
                        }
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                    report["total_pages_extracted"] += 1
                
                if not has_text:
                    report["failed_files"].append({
                        "path": str(pdf_path),
                        "reason": "No extractable text found in any page"
                    })
                else:
                    report["files_processed"] += 1
                    
            except Exception as e:
                report["failed_files"].append({
                    "path": str(pdf_path),
                    "reason": f"Error: {str(e)}"
                })
                print(f"Failed to process {pdf_path.name}: {e}")
                
    with open(REPORT_FILE, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    print(f"Pipeline finished. Processed {report['files_processed']} files, extracted {report['total_pages_extracted']} pages.")
    print(f"Skipped {report['duplicates_skipped']} duplicates.")
    print(f"Failed {len(report['failed_files'])} files.")

if __name__ == "__main__":
    run_pipeline()
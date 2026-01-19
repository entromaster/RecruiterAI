import fitz
import json
import re
from pathlib import Path
from typing import Union, Dict, List, Any, Tuple

class PDFProcessor:
    def __init__(self, pdf_path, json_file, regex_patterns: Union[None, re.Pattern, List[re.Pattern]], images_folder="./pages", padding=5):
        self.pdf_path = pdf_path
        self.json_file = json_file
        # Convert single pattern to list or keep as None
        self.regex_patterns = [regex_patterns] if isinstance(regex_patterns, re.Pattern) else regex_patterns
        self.padding = padding
        self.pymupdf_bboxes = {}
        self.adobe_bboxes_by_page = {}

    def get_pymupdf_bboxes(self, min_width=2.0, min_text_len=1):
        """Extract bounding boxes from PDF pages using PyMuPDF."""
        try:
            doc = fitz.open(self.pdf_path)
        except Exception as e:
            raise FileNotFoundError(f"Could not open PDF file at '{self.pdf_path}': {e}")

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                blocks = page.get_text("dict")["blocks"]

                # Get page dimensions at 300 DPI
                dpi = 300
                scale_factor = dpi / 72  # Convert from PDF points (72 DPI) to desired DPI
                page_width = page.rect.width * scale_factor
                page_height = page.rect.height * scale_factor

                bboxes_per_page = []

                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                bbox = span["bbox"]
                                text = span["text"].strip()
                                box_width = bbox[2] - bbox[0]
                                if box_width < min_width or len(text) < min_text_len or text.isspace():
                                    continue
                                x0, y0, x1, y1 = [int(coord * scale_factor) for coord in bbox]
                                bboxes_per_page.append({"bbox": [x0, y0, x1, y1], "text": text})

                self.pymupdf_bboxes[page_num] = {
                    "dimensions": {"width": int(page_width), "height": int(page_height)},
                    "bboxes": bboxes_per_page,
                    "scale_factor": scale_factor
                }
        except Exception as e:
            raise RuntimeError(f"Error processing PDF pages for bounding boxes: {e}")
        finally:
            doc.close()

    def load_json_data(self):
        try:
            with open(self.json_file, "r") as file:
                data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {self.json_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in file '{self.json_file}': {e}")

        if not all(key in data for key in ["elements", "pages"]):
            raise KeyError("JSON file must contain 'elements' and 'pages' keys.")

        return data["elements"], data["pages"]

    def organize_by_page(self, elements):
        organized = {}
        for element in elements:
            page = element.get("Page", 0)
            bounds = element.get("Bounds", [])
            organized.setdefault(page, []).append(bounds)
        return organized

    def pad_bbox(self, bbox, img_width, img_height):
        x1, y1, x2, y2 = bbox
        return [
            max(0, x1 - self.padding),
            max(0, y1 - self.padding),
            min(img_width - 1, x2 + self.padding),
            min(img_height - 1, y2 + self.padding)
        ]

    def get_adobe_bboxes(self):
        try:
            elements, pages_info = self.load_json_data()
            organized_data = self.organize_by_page(elements)
        except Exception as e:
            raise RuntimeError(f"Error loading JSON data: {e}")

        try:
            for page_number in self.pymupdf_bboxes.keys():
                if page_number in organized_data and page_number < len(pages_info):
                    page_dimensions = self.pymupdf_bboxes[page_number]["dimensions"]
                    img_width, img_height = page_dimensions["width"], page_dimensions["height"]

                    bboxes = organized_data[page_number]
                    pdf_width = pages_info[page_number]["width"]
                    pdf_height = pages_info[page_number]["height"]

                    adjusted_bboxes = self.adjust_bboxes_to_image(
                        bboxes, pdf_width, pdf_height, img_width, img_height
                    )

                    padded_bboxes = [
                        self.pad_bbox(bbox, img_width, img_height)
                        for bbox in adjusted_bboxes
                    ]

                    self.adobe_bboxes_by_page[page_number] = {
                        "dimensions": page_dimensions,
                        "bboxes": padded_bboxes
                    }
        except Exception as e:
            raise RuntimeError(f"Error processing Adobe bounding boxes: {e}")

    def adjust_bboxes_to_image(self, bboxes, pdf_width, pdf_height, img_width, img_height):
        scale_x, scale_y = img_width / pdf_width, img_height / pdf_height
        adjusted_bboxes = []
        for bbox in bboxes:
            if len(bbox) == 4:
                left, bottom, right, top = bbox
                x1 = int(left * scale_x)
                y1 = int((pdf_height - top) * scale_y)
                x2 = int(right * scale_x)
                y2 = int((pdf_height - bottom) * scale_y)

                adjusted_bboxes.append([
                    max(0, min(x1, img_width - 1)),
                    max(0, min(y1, img_height - 1)),
                    max(0, min(x2, img_width - 1)),
                    max(0, min(y2, img_height - 1))
                ])
        return adjusted_bboxes

    def bbox_intersects(self, boxA, boxB):
        x1A, y1A, x2A, y2A = boxA
        x1B, y1B, x2B, y2B = boxB
        return not (x2A <= x1B or x2B <= x1A or y2A <= y1B or y2B <= y1A)

    def find_non_intersecting_bboxes(self):
        missed_text = {}
        try:
            for page_num, page_data in self.pymupdf_bboxes.items():
                pymupdf_bboxes_per_page = page_data["bboxes"]
                adobe_bboxes_per_page = self.adobe_bboxes_by_page.get(page_num, {}).get("bboxes", [])

                remaining_bboxes = [
                    entry for entry in pymupdf_bboxes_per_page
                    if not any(self.bbox_intersects(entry["bbox"], adobe_bbox)
                              for adobe_bbox in adobe_bboxes_per_page)
                ]

                missed_text[page_num] = " ".join(item['text'] for item in remaining_bboxes)
                self.pymupdf_bboxes[page_num]["remaining_bboxes"] = remaining_bboxes

        except Exception as e:
            raise RuntimeError(f"Error finding non-intersecting bounding boxes: {e}")

        return missed_text

    def extract_footer_text(self):
        """
        Attempts to extract footer text using an adaptive approach:
        - If bottom 35% of page is empty, uses bottommost text as reference
        - Otherwise uses standard page height method
        Updates self.pymupdf_bboxes with the footer text and bounding boxes.
        Returns a dictionary with page numbers as keys and extracted footer text as values.
        """
        doc = fitz.open(self.pdf_path)
        footer_text = {}

        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_height = page.rect.height
                page_width = page.rect.width

                # Initialize footer_bboxes for this page
                footer_bboxes = []

                # Check if bottom 35% is empty
                check_height = page_height * 0.65  # Starting point for empty space check
                bottom_check = page.get_text("dict", clip=(0, check_height, page_width, page_height))["blocks"]
                
                if not bottom_check:  # Bottom 35% is empty
                    # Find bottommost text block
                    blocks = page.get_text("dict")["blocks"]
                    max_bottom = 0
                    if blocks:
                        for block in blocks:
                            if "lines" in block:
                                block_bottom = block["bbox"][3]  # y1 coordinate
                                max_bottom = max(max_bottom, block_bottom)
                    
                    if max_bottom > 0:
                        # Calculate footer region as 20% up from bottommost text
                        percentage = 20
                        footer_height = max_bottom * (percentage / 100)
                        start_height = max_bottom - footer_height
                    else:
                        # No text found, set empty footer
                        footer_text[page_num] = ""
                        continue
                else:
                    # Use original method - 20% of page height
                    percentage = 20
                    start_height = page_height * (1 - percentage / 100)

                # Get text blocks from the footer region
                blocks = page.get_text("dict", clip=(0, start_height, page_width, page_height))["blocks"]

                # Extract text and bounding boxes
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    # Scale the bbox coordinates to match the 300 DPI scale factor
                                    scale_factor = self.pymupdf_bboxes[page_num]["scale_factor"]
                                    bbox = span["bbox"]
                                    scaled_bbox = [
                                        int(bbox[0] * scale_factor),
                                        int(bbox[1] * scale_factor),
                                        int(bbox[2] * scale_factor),
                                        int(bbox[3] * scale_factor),
                                    ]

                                    footer_bboxes.append({"bbox": scaled_bbox, "text": text})

                # Store the footer text
                footer_text[page_num] = " ".join(item["text"] for item in footer_bboxes)

                # Append the footer boxes to 'remaining_bboxes'
                if "remaining_bboxes" in self.pymupdf_bboxes[page_num]:
                    self.pymupdf_bboxes[page_num]["remaining_bboxes"].extend(footer_bboxes)
                else:
                    self.pymupdf_bboxes[page_num]["remaining_bboxes"] = footer_bboxes

        except Exception as e:
            print(f"Error extracting footer text: {str(e)}")
            # Return empty strings for all pages in case of error
            footer_text = {page_num: "" for page_num in range(len(doc))}

        finally:
            doc.close()

        return footer_text

    def match_text_with_regex(self):
        """
        Match text against multiple regex patterns.
        Returns matches for each page with the first successful pattern match.
        """
        if not self.regex_patterns:
            return {}

        all_matches = []

        # For each text entry, try each pattern until one matches
        for page_num, page_data in self.pymupdf_bboxes.items():
            for entry in page_data.get("remaining_bboxes", []):
                for pattern in self.regex_patterns:
                    if pattern.match(entry["text"]):
                        all_matches.append({
                            "page_num": page_num,
                            "bbox": entry["bbox"],
                            "text": entry["text"]
                        })
                        break  # Stop trying patterns once we find a match

        # Group matches by page number and remove duplicates
        matches_to_keep = {}
        for i in range(len(all_matches)):
            current_match = all_matches[i]
            if i == len(all_matches) - 1 or current_match["text"] != all_matches[i + 1]["text"]:
                page_num = current_match["page_num"]
                matches_to_keep.setdefault(page_num, []).append(current_match)

        return matches_to_keep

    def process(self):
        try:
            self.get_pymupdf_bboxes()
            self.get_adobe_bboxes()
            missed_text = self.find_non_intersecting_bboxes()

            #print("Missed text by Adobe API is: ", missed_text)

            # Always extract the bottom 20% of text from the PDF
            footer_text = self.extract_footer_text()

            #print(self.pymupdf_bboxes)

            # Merge the missed_text with footer_text
            for page_num in self.pymupdf_bboxes.keys():
                missed = missed_text.get(page_num, "")
                footer = footer_text.get(page_num, "")
                if footer:
                    if missed:
                        missed_text[page_num] = f"{missed} {footer}"
                    else:
                        missed_text[page_num] = footer

            if not self.regex_patterns:
                return missed_text

            return self.match_text_with_regex(), missed_text
        except Exception as e:
            raise RuntimeError(f"Error during PDF processing: {e}")


class DocumentSplitter:
    def __init__(self, document_id: str, json_path: str, pdf_path: str):
        """
        Initialize DocumentSplitter with document ID, JSON path, and PDF path.
        """
        self.json_path = json_path
        self.document_id = document_id
        self.pdf_path = pdf_path
        self.structured_json = self.load_structured_json()
        self.structured_data = self.structured_json.get('elements', [])
        self.pdf_info = self.get_pdf_info()

    def load_structured_json(self) -> Dict[str, Any]:
        """
        Load and return the structured JSON data from file.
        """
        try:
            with open(self.json_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found at path: {self.json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON file at {self.json_path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error while loading JSON file: {e}")

    def get_pdf_info(self) -> Dict[int, Dict[str, float]]:
        """
        Get PDF page information including dimensions and scale factors.
        """
        pdf_info = {}
        try:
            doc = fitz.open(self.pdf_path)
            for page_num in range(len(doc)):
                page = doc[page_num]
                pdf_width = page.rect.width
                pdf_height = page.rect.height
                pdf_info[page_num] = {
                    "pdf_width": pdf_width,
                    "pdf_height": pdf_height
                }
            doc.close()
            return pdf_info
        except FileNotFoundError:
            raise FileNotFoundError(f"PDF file not found at path: {self.pdf_path}")
        except Exception as e:
            raise RuntimeError(f"Error processing PDF file at {self.pdf_path}: {e}")

    def get_page_dimensions(self, page_num: str) -> Tuple[float, float, float, float]:
        """
        Get both PDF and scaled dimensions for a specific page.
        Returns: (pdf_width, pdf_height, scaled_width, scaled_height)
        """
        try:
            page_int = int(page_num)
        except ValueError:
            raise ValueError(f"Invalid page number format: {page_num}")

        page_info = self.pdf_info.get(page_int, {})
        if not page_info:
            raise KeyError(f"Page {page_int} not found in PDF info.")

        pdf_width = page_info.get("pdf_width", 612.0)
        pdf_height = page_info.get("pdf_height", 792.0)

        scale_x = scale_y = 1.0
        for page in self.structured_json.get('pages', []):
            if page.get('page_number') == page_int:
                scale_x = page.get('width', pdf_width) / pdf_width
                scale_y = page.get('height', pdf_height) / pdf_height
                break

        scaled_width = pdf_width * scale_x
        scaled_height = pdf_height * scale_y

        return pdf_width, pdf_height, scaled_width, scaled_height

    def reverse_bbox_adjustment(self, bbox: List[int], scaled_width: float, scaled_height: float,
                              pdf_width: float, pdf_height: float) -> List[int]:
        """
        Convert bbox from scaled coordinates back to PDF coordinates.
        """
        try:
            scale_x = pdf_width / scaled_width
            scale_y = pdf_height / scaled_height
            x1, y1, x2, y2 = bbox

            left = int(x1 * scale_x)
            right = int(x2 * scale_x)
            top = int((scaled_height - y1) * scale_y)
            bottom = int((scaled_height - y2) * scale_y)

            return [left, bottom, right, top]
        except Exception as e:
            raise ValueError(f"Error adjusting bounding box {bbox}: {e}")

    def organize_global_indices(self) -> Dict[str, List[Tuple[int, Dict[str, Any]]]]:
        """
        Organize structured data by page number and assign global indices to each element.
        """
        if not self.structured_data:
            raise ValueError("No structured data found in JSON file.")

        page_organized = {}
        for global_idx, item in enumerate(self.structured_data):
            if 'Page' in item:
                page_num = item['Page']
                if page_num not in page_organized:
                    page_organized[page_num] = []
                page_organized[page_num].append((global_idx, item))
        return page_organized

    def find_split_indices_global(self, match_dict: Dict[str, List[Dict[str, Any]]]) -> List[int]:
        """
        Find the split indices for each page based on footer positions, using global indices.
        """
        try:
            organized_data = self.organize_global_indices()
        except ValueError as e:
            raise ValueError(f"Error organizing global indices: {e}")

        split_indices = []

        for page_num, matches in match_dict.items():
            if page_num not in organized_data:
                continue

            try:
                pdf_width, pdf_height, scaled_width, scaled_height = self.get_page_dimensions(page_num)
            except Exception as e:
                raise RuntimeError(f"Error getting page dimensions for page {page_num}: {e}")

            page_elements = organized_data[page_num]

            for match in matches:
                try:
                    footer_bbox = self.reverse_bbox_adjustment(
                        match['bbox'],
                        scaled_width,
                        scaled_height,
                        pdf_width,
                        pdf_height
                    )

                    footer_y = footer_bbox[1]  # Bottom coordinate of the footer

                    elements_below = [
                        global_idx for global_idx, element in page_elements
                        if element.get('Bounds', [0, 0, 0, 0])[1] > footer_y
                    ]

                    if not elements_below:
                        split_indices.append(page_elements[-1][0])
                    else:
                        split_indices.append(min(elements_below) - 1)
                except Exception as e:
                    raise RuntimeError(f"Error finding split indices on page {page_num}: {e}")

        return sorted(set(split_indices))

    def process_document_splits(self, match_dict: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
      """
      Process document splits and return a dictionary of split info, including:
          - (start, end) element indices
          - (min_page, max_page) coverage
      """
      if not self.structured_data:
          raise ValueError("No elements found in structured JSON.")

      try:
          split_indices = self.find_split_indices_global(match_dict)
      except Exception as e:
          raise RuntimeError(f"Error processing split indices: {e}")

      indexed_splits = {}
      for idx in range(len(split_indices)):
          start = 0 if idx == 0 else split_indices[idx - 1] + 1
          end = split_indices[idx]

          # Avoid an empty slice
          if start == end:
              end += 1

          # Gather all pages covered by elements [start..end]
          pages_in_split = set()
          for element_idx in range(start, end + 1):
              element = self.structured_data[element_idx]
              pages_in_split.add(element.get("Page", 0))

          # Find min & max page covered
          if pages_in_split:
              min_page = min(pages_in_split)
              max_page = max(pages_in_split)
          else:
              # Fallback if somehow empty
              min_page = max_page = 0

          indexed_splits[f"{self.document_id}_{idx}"] = {
              "split": (start, end),
              "page_range": (min_page, max_page),  # store multi-page coverage
          }

      return indexed_splits



class DocumentSplitProcessor:
    def __init__(self, document_id: str, pdf_path: str, json_file: str,
                 regex_patterns: Union[None, re.Pattern, List[re.Pattern]] = None,
                 images_folder: str = "./pages"):
        self.pdf_path = pdf_path
        self.json_file = json_file
        self.regex_patterns = regex_patterns  # Can be None, single pattern, or list of patterns
        self.images_folder = images_folder
        self.document_id = document_id

    def run(self) -> Tuple[Union[Dict, List], str]:
      try:
          processor = PDFProcessor(self.pdf_path, self.json_file, self.regex_patterns, self.images_folder)

          if self.regex_patterns is not None:
              splitter = DocumentSplitter(self.document_id, self.json_file, self.pdf_path)

              # Process the PDF to get matches (and missed_text)
              matches, missed_text = processor.process()
              print(f"Missed Text: {missed_text}")
              print(f"Matches: {matches}")

              # Find split indices (start, end) + page ranges
              split_indices = splitter.process_document_splits(matches)
              print("Split indices: ", split_indices)

              # Now build an output dict, using the first item in each match's value
              # Sort the page numbers so you can consume them in order
              sorted_pages = sorted(matches.keys())

              # Extract the first matched text from each page
              # (Adjust if you prefer merging all texts on a page, or a different selection)
              page_texts = []
              for pg in sorted_pages:
                  if not matches[pg]:
                      continue
                  # Take the first match’s text from that page
                  first_match_text = matches[pg][0]['text']
                  page_texts.append((pg, first_match_text))

              print("Page Texts:", page_texts)
              print("length of page texts:", len(page_texts))

              # Build the final result by iterating over split_indices keys and
              # assigning the "page_num" from the `page_texts` array in sequence.
              output_dict = {}
              idx = 0
              firstpageval = 0

              # Ensure we have enough page_texts for each split key
              # (If your logic demands a 1:1 mapping, you may want to handle
              #  the case where you run out of page_texts or have leftover splits differently.)
              for key, split_data in split_indices.items():
                  print(key, split_data)
                  print(page_texts[idx][1])
                  if idx < len(page_texts):
                      page_num, text = page_texts[idx]
                  else:
                      # Fallback if there are more splits than page texts
                      page_num, text = (0, "")

                  output_dict[key] = {
                      "text": text,
                      "split": split_data["split"],  # e.g. (start, end)
                      "page_num": (firstpageval, page_num)  # from the matched page
                  }
                  firstpageval =page_num+1
                  idx += 1

              # Optionally, merge duplicates if needed
              merged_output = self.merge_similar_entries(output_dict)

              return merged_output, "Success"

          else:
              # No regex patterns => just get missed text
              missed_text = processor.process()
              return missed_text, "Success"

      except Exception as e:
          return (), f"Unexpected error in DocumentSplitProcessor.run: {e}"



    def merge_similar_entries(self, input_dict: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
      """
      Merges dictionary entries that have the same 'text' value
      and combines their (start,end) splits plus (min_page,max_page).
      """
      try:
          # Example: parse the prefix from the first key
          first_key = next(iter(input_dict), None)
          prefix = first_key.rsplit('_', 1)[0] if first_key else "doc"

          text_groups = {}

          # 1. Group by text
          for key, value in input_dict.items():
              text = value["text"]
              if text not in text_groups:
                  text_groups[text] = {
                      "keys": [key],
                      "splits": [value["split"]],          # list of (start, end)
                      "page_ranges": [value["page_num"]],  # list of (min_page, max_page)
                  }
              else:
                  text_groups[text]["keys"].append(key)
                  text_groups[text]["splits"].append(value["split"])
                  text_groups[text]["page_ranges"].append(value["page_num"])

          # 2. Create merged entries
          output_dict = {}
          index = 0
          for text, group_data in text_groups.items():
              all_splits = group_data["splits"]
              all_page_ranges = group_data["page_ranges"]

              # Merge splits
              min_split = min(s[0] for s in all_splits)
              max_split = max(s[1] for s in all_splits)

              # Merge page ranges
              min_page = min(r[0] for r in all_page_ranges)
              max_page = max(r[1] for r in all_page_ranges)

              merged_key = f"{prefix}_{index}"
              output_dict[merged_key] = {
                  "text": text,
                  "split": (min_split, max_split),
                  "page_num": (min_page, max_page),
              }
              index += 1

          return output_dict

      except Exception as e:
          print(f"Error merging similar entries: {e}")
          raise


# Example usage:
if __name__ == "__main__":
    pdf_path = "path/to/your.pdf"
    json_file = "path/to/your.json"
    images_folder = "pages"
    
    # Example patterns
    pattern1 = re.compile(r'^[A-Z]+-\d{3}$')
    pattern2 = re.compile(r'^[A-Z]+-[A-Z]+-\d{3}$')
    
    # You can now use it in multiple ways:
    # 1. With a single pattern
    processor1 = DocumentSplitProcessor("123", pdf_path, json_file, pattern1)
    
    # 2. With multiple patterns
    processor2 = DocumentSplitProcessor("123", pdf_path, json_file, [pattern1, pattern2])
    
    # 3. With no pattern
    processor3 = DocumentSplitProcessor("123", pdf_path, json_file, None)
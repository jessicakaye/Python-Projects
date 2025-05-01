"""
This is used to generate an Originals report based on a PDF export from a Dashboard and a Slides file. It will search through the base folder in a shared drive.

Logic Order:
Initialize Python libraries
Find Dashboard PDF then Crop
Find Slides doc then convert to PDF
Combine the 2 PDFs, then reorganize
"""



#@title Libraries

!pip install -q --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib PyPDF2
!pip install -q --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os
import re
from copy import deepcopy
from PyPDF2 import PdfReader, PdfWriter
import ipywidgets as widgets
from IPython.display import display, clear_output

from google.colab import drive
drive.mount('/content/drive')

from google.colab import auth
auth.authenticate_user()



#@title Main Functions & Setup

# === Constants ===
SHARED_DRIVE_BASE = '/content/drive/Shareddrives/'
PAGE_WIDTH = 975
# Shared Drive and Parent Folder IDs
SHARED_DRIVE_ID = ''  # Replace with your actual Shared Drive ID
PARENT_FOLDER_ID = ''  # Replace with your actual Parent Folder ID

# === UI Elements ===
dropdown = widgets.Dropdown(
    options=["7 days", "14 days", "28 days", "90 days"],
    description="Select:",
)

folder_input = widgets.Text(
    value='',
    placeholder='e.g. Team Drive/Reports/April',
    description='Folder:',
    layout=widgets.Layout(width='600px')
)

select_button = widgets.Button(
    description="Search PDFs",
    style={'button_color': 'green', 'font_weight': 'bold', 'font_size': '14px'}
)

# Radio button for Movie or Series
media_type_radio = widgets.RadioButtons(
    options=['Movie', 'Series'],
    description='Media Type:',
    disabled=False
)

# Radio button for Returning Season (only visible if Series is selected)
returning_season_radio = widgets.RadioButtons(
    options=['Yes', 'No'],
    description='Returning Season:',
    disabled=False,
    layout=widgets.Layout(width='auto'),
    style={'description_width': 'initial'}
)

crop_button = widgets.Button(
    description="Crop Selected PDF",
    style={'button_color': 'green', 'font_weight': 'bold', 'font_size': '14px'}
)
crop_button.layout.display = 'none'  # <-- initially hidden

file_selector = widgets.Dropdown(
    options=[],
    description='Select File:',
    layout=widgets.Layout(width='600px')
)
file_selector.layout.display = 'none'  # <-- initially hidden

merge_button = widgets.Button(
    description="Merge PDFs",
    style={'button_color': 'green', 'font_weight': 'bold', 'font_size': '14px'}
)
merge_button.layout.display = 'none'  # <-- initially hidden

progress_bar = widgets.IntProgress(
    value=0,
    min=0,
    max=100,
    description='Cropping:',
    bar_style='info',
    style={'description_width': 'initial'}
)
progress_bar.layout.display = 'none'  # <-- initially hidden

slides_selector = widgets.Dropdown(
    options=[],
    description='Slides:',
    layout=widgets.Layout(width='600px')
)
slides_selector.layout.display = 'none'  # <-- initially hidden



# === Select Button Click ===
def on_select_click(b):
    clear_output(wait=True)

    # Show the folder input and file search button
    display(dropdown, folder_input, media_type_radio, returning_season_radio)
    # Show the first buttons
    display(select_button)

    relative_path = folder_input.value.strip()
    folder_path = os.path.join(SHARED_DRIVE_BASE, relative_path)

    if not os.path.exists(folder_path):
        print("\n ❌ Folder not found.")
        file_selector.layout.display = 'none'
        crop_button.layout.display = 'none'
        return

    # Show file search button and proceed
    selected = dropdown.value.split()[0]
    pattern = re.compile(rf'{selected}\s*days.*dashboard', re.IGNORECASE)

    matched_files = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(".pdf") and pattern.search(f)]

    if not matched_files:
        print("\n ❌ No matching PDFs found.")
        return

    file_selector.options = matched_files
    file_selector.value = matched_files[0]

    if len(matched_files) == 1:
        print(f"\n\n 📄 One matching file found: {matched_files[0]}")
    else:
        print(f"\n\n 📄 Found {len(matched_files)} matching files. Please select one.")

        # Make them visible first
    file_selector.layout.display = None
    crop_button.layout.display = None

    # Now display them
    display(file_selector)
    # print(f"\n")
    display(crop_button)


# === Crop Function with Dynamic Dimensions ===
def crop_top_and_bottom(file_path, output_path, option, is_series, is_returning_season):
    reader = PdfReader(file_path)
    writer = PdfWriter()

    if option == "7":
      PAGE_HEIGHT = 3150
    else:
      PAGE_HEIGHT = 3750


    num_pages = len(reader.pages)
    progress_bar.max = num_pages * (2 if not is_series else 1)
    progress_bar.value = 0
    progress_bar.layout.display = 'block'

    for page in reader.pages:
        if is_series:
            # Adjust dimensions for series
            if is_returning_season:
                # Returning season: slightly different crop
                crop_box = (0,370)
            else:
                # New season
                crop_box = (0, 880)

            if option == "7":
              crop_box = crop_box[0],crop_box[1]-60

            # Only one crop
            cropped_page = deepcopy(page)
            cropped_page.mediabox.lower_left = crop_box
            cropped_page.mediabox.upper_right = (PAGE_WIDTH, PAGE_HEIGHT)
            writer.add_page(cropped_page)
            progress_bar.value += 1

        else:
            # MOVIE: split into top and bottom parts
            top_crop = (880, PAGE_HEIGHT)
            bottom_crop = (0, 366)

            if option == "7":
              top_crop = top_crop[0]-60,top_crop[1]
              bottom_crop = bottom_crop[0],bottom_crop[1]-60

            # Top part
            top_page = deepcopy(page)
            top_page.mediabox.lower_left = (0, top_crop[0])
            top_page.mediabox.upper_right = (PAGE_WIDTH, top_crop[1])
            writer.add_page(top_page)
            progress_bar.value += 1

            # Bottom part
            bottom_page = deepcopy(page)
            bottom_page.mediabox.lower_left = (0, bottom_crop[0])
            bottom_page.mediabox.upper_right = (PAGE_WIDTH, bottom_crop[1])
            writer.add_page(bottom_page)
            progress_bar.value += 1

    with open(output_path, "wb") as f:
        writer.write(f)

    progress_bar.layout.display = 'none'
    print(f"\n ✅ Cropped PDF saved at: {output_path}")



# === Crop Button Click ===
def on_crop_click(b):
    relative_path = folder_input.value.strip()
    folder_path = os.path.join(SHARED_DRIVE_BASE, relative_path)

    selected_file = file_selector.value
    if not selected_file:
        print("\n ❌ No file selected.")
        return

    file_path = os.path.join(folder_path, selected_file)
    option = dropdown.value.split()[0]

    # Get the media type (movie or series) and whether it's a returning season
    is_series = media_type_radio.value == 'Series'
    is_returning_season = returning_season_radio.value == 'Yes' if is_series else False

    # Include folder name in output filename
    folder_name_part = os.path.basename(folder_path)
    folder_name_part = folder_name_part.replace(" ", "_")  # replace spaces (optional)

    output_filename = f"{folder_name_part}_{option}_days_dashboard_cropped.pdf"
    output_path = os.path.join(folder_path, output_filename)

    if os.path.exists(output_path):
        print(f"\n ⚠️ File '{output_filename}' already exists.\n")
        # Show options to continue or cancel
        continue_button = widgets.Button(
            description="Continue",
            style={'button_color': 'green', 'font_weight': 'bold', 'font_size': '14px'}
        )
        cancel_button = widgets.Button(
            description="Cancel",
            style={'button_color': 'red', 'font_weight': 'bold', 'font_size': '14px'}
        )

        # Display the confirmation buttons
        display(continue_button, cancel_button)

        def on_continue_click(c):
            print("\n\n ✅ Proceeding to search for slides.")
            parent_folder_id = search_folder_id_in_shared_drive(SHARED_DRIVE_ID, folder_input.value.strip())
            if parent_folder_id:
                search_slides_in_folder(SHARED_DRIVE_ID, parent_folder_id, option)

        def on_cancel_click(c):
            print("\n\n ❌ Operation canceled.")
            clear_output()

        continue_button.on_click(on_continue_click)
        cancel_button.on_click(on_cancel_click)

    else:
        crop_top_and_bottom(file_path, output_path, option, is_series, is_returning_season)
        # After cropping, proceed to search for slides
        parent_folder_id = search_folder_id_in_shared_drive(SHARED_DRIVE_ID, folder_input.value.strip())
        if parent_folder_id:
            search_slides_in_folder(SHARED_DRIVE_ID, parent_folder_id, option)



# Function to search for a folder by name and get the folder ID
def search_folder_id_in_shared_drive(shared_drive_id, search_query=None):
    try:
        # Build the Google Drive API client
        drive_service = build('drive', 'v3')

        # Base query to search for folders in the shared drive
        query = f"mimeType='application/vnd.google-apps.folder'"

        # If a search query (folder name) is provided, append it to the filter
        if search_query:
            query += f" and name contains '{search_query}'"

        # Execute the search query
        results = drive_service.files().list(
            q=query,
            driveId=shared_drive_id,  # Shared Drive ID
            corpora='drive',  # Limit search to the Shared Drive
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields="files(id, name)"
        ).execute()

        items = results.get('files', [])

        if not items:
            print(f'No folder found with name containing "{search_query}" in Shared Drive.')
            return None
        else:
            # for item in items:
            #     # print(f"Folder Name: {item['name']}, Folder ID: {item['id']}")
            return items[0]['id']  # Return the ID of the first matching folder (assuming there's only one match)

    except HttpError as error:
        print(f"An error occurred: {error}")
        return None

# Function to search for a Google Slides document based on the dropdown value
def search_slides_in_folder(shared_drive_id, folder_id, option):
    try:
        # Build the Google Drive API client
        drive_service = build('drive', 'v3')

        # Create the search query based on the dropdown option and "highlights"
        search_query = f"{option.replace(' ', '%')}%highlights"  # Replace spaces with wildcards

        # Base query to search for Google Slides within the folder
        query = f"'{folder_id}' in parents and mimeType='application/vnd.google-apps.presentation' and name contains '{search_query}'"

        # Execute the search query
        results = drive_service.files().list(
            q=query,
            driveId=shared_drive_id,
            corpora='drive',
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            fields="files(id, name)"
        ).execute()

        items = results.get('files', [])

        # If slides are found, update the dropdown options and display it
        if items:
            slide_options = [(item['name'], item['id']) for item in items]
            slides_selector.options = slide_options  # update options
            print(f"\nFound {len(items)} slides. Please select one.")
            slides_selector.layout.display = 'inline-block'  # make visible
            display(slides_selector)  # <-- YOU MUST re-display it to show the updated widget

          # SHOW MERGE BUTTON
            merge_button.layout.display = 'inline-block'
            display(merge_button)
        else:
            print(f'No Google Slides found matching "{search_query}" in folder {folder_id}.')


    except HttpError as error:
        print(f"An error occurred: {error}")

# Export Slides to PDF
def export_slides_to_pdf(slides_id, save_path):
    drive_service = build('drive', 'v3')
    request = drive_service.files().export_media(
        fileId=slides_id,
        mimeType='application/pdf'
    )
    with open(save_path, 'wb') as f:
        f.write(request.execute())
    print(f"\n ✅ Slides exported to PDF at: {save_path}")

# Merge: Slides first, then Cropped PDF
def merge_and_rearrange_pdfs(slide_pdf_path, cropped_pdf_path, output_path):
    slide_reader = PdfReader(slide_pdf_path)
    cropped_reader = PdfReader(cropped_pdf_path)
    writer = PdfWriter()

    # Add all slides pages first
    for page in slide_reader.pages:
        writer.add_page(page)

    # Insert cropped pages AFTER page 2
    if len(writer.pages) >= 2:
        # Save first two pages separately
        page1 = writer.pages[0]
        page2 = writer.pages[1]
        rest = writer.pages[2:]

        # Start new writer
        new_writer = PdfWriter()
        new_writer.add_page(page1)
        new_writer.add_page(page2)

        for page in cropped_reader.pages:
            new_writer.add_page(page)

        for page in rest:
            new_writer.add_page(page)

        # Save merged PDF
        with open(output_path, "wb") as f:
            new_writer.write(f)

    else:
        print("❌ Slides PDF does not have at least 2 pages.")


# === Merge Button Click ===
def on_merge_click(b):
    clear_output(wait=True)
    display(dropdown, folder_input, media_type_radio, returning_season_radio, select_button, file_selector, crop_button, progress_bar, slides_selector, merge_button)

    relative_path = folder_input.value.strip()
    folder_path = os.path.join(SHARED_DRIVE_BASE, relative_path)

    slide_id = slides_selector.value
    if not slide_id:
        print("\n❌ No slide selected.")
        return

    option = dropdown.value.split()[0]

    folder_name_part = os.path.basename(folder_path).replace(" ", "_")
    cropped_pdf_filename = f"{folder_name_part}_{option}_days_dashboard_cropped.pdf"
    cropped_pdf_path = os.path.join(folder_path, cropped_pdf_filename)

    if not os.path.exists(cropped_pdf_path):
        print(f"\n❌ Cropped PDF '{cropped_pdf_filename}' not found!")
        return

    # --- Export Slides ---
    drive_service = build('drive', 'v3')
    try:
        export_request = drive_service.files().export_media(
            fileId=slide_id,
            mimeType='application/pdf'
        )
        temp_slide_pdf_path = os.path.join(folder_path, "temp_slide_export.pdf")
        with open(temp_slide_pdf_path, 'wb') as f:
            f.write(export_request.execute())
    except HttpError as error:
        print(f"❌ An error occurred during export: {error}")
        return

    # --- Merge and Rearrange ---
    final_output_path = os.path.join(folder_path, f"{os.path.basename(folder_path)} {option} Day Perf US 2025.pdf")
    merge_and_rearrange_pdfs(temp_slide_pdf_path, cropped_pdf_path, final_output_path)

    # --- Delete Temp Slide PDF ---
    try:
        os.remove(temp_slide_pdf_path)
    except Exception as e:
        print(f"⚠️ Could not delete temp file: {e}")

    # --- Final Clean Output ---
    print(f"\n✅ Final merged PDF created:\n{final_output_path}")


# === Bind Buttons ===
select_button.on_click(on_select_click)
crop_button.on_click(on_crop_click)
merge_button.on_click(on_merge_click)

#@title Display Input Options

# === Display UI ===
display(dropdown, folder_input, media_type_radio, returning_season_radio, select_button, file_selector, crop_button, progress_bar, slides_selector, merge_button)

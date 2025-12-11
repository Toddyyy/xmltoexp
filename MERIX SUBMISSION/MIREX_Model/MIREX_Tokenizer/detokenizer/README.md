### **Detokenizer Usage Guide**

#### **1. Introduction**

This project contains a `Detokenizer` tool whose core function is to render a JSON file containing score information and AI model performance predictions into a standard, playable MIDI (`.mid`) file.

This guide will walk you through the complete process from environment setup to successfully generating a MIDI file.

#### **2. Environment Setup**

Before running the script, please strictly follow the steps below to configure your Python environment.

**A. Python Version:**
*   Please ensure you have Python 3.12 installed.

**B. Create and Activate a Virtual Environment:**
*   To avoid conflicts with other Python projects on your system, creating an isolated virtual environment is strongly recommended. In your terminal, navigate to the root directory of the `MIREX` project, then run:

    *   **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    *   **macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   After successful activation, your terminal prompt should be prefixed with a `(venv)` tag.

**C. Install All Dependencies using `requirements.txt`:**
*   Your project includes a `requirements.txt` file, which lists all the required libraries and their exact versions for this project.
*   Please ensure your virtual environment is activated, then from the project's root directory, run this **single command**:

    ```bash
    pip install -r requirements.txt
    ```
*   `pip` will automatically read this file and install all necessary libraries for you (such as `numpy`, `scipy`, `mido`, `partitura`, etc.).

**D. Confirm Project File Structure:**
*   Please ensure your folder structure is correct. All commands should be run from the **project's root directory** (e.g., `MIREX/`).
    ```
    MIREX/
    ├── detokenizer/
    │   ├── detokenizer.py
    │   └── run_detokenizer.py  <-- This is the script we will run
    │
    ├── requirements.txt      <-- The file for installing dependencies
    │
    ├── example_data/
    │   └── ...
    │
    └── tokenizer/
        └── ...
    ```

#### **3. How to Run**

This `Detokenizer` is executed via the `run_detokenizer.py` script. It is a command-line tool that accepts two required arguments.

**Command Format:**
```bash
python -m detokenizer.run_detokenizer [path_to_input_json] [path_for_output_midi]
```

**Viewing Help:**
If you are unsure how to use it, you can run the following command at any time to see a detailed description of all options:
```bash
python detokenizer/run_detokenizer.py --help
```

#### **4. Full Run Example**

Now, let's run a complete example.

1.  **Open your terminal**.
2.  **Ensure** you have **activated the virtual environment** `(venv)` and that your current directory is the project's **root directory**, `MIREX/`.
3.  **Copy and run** the following command:

    ```bash
    python detokenizer/run_detokenizer.py "example_data/detokenizer/Aug_16_Beethoven_32_Variations_in_C_minor_WoO_80.json" "beethoven_final_performance.mid"
    ```

#### **5. Expected Output**

After a successful run, you will see a log in your terminal similar to the one below, ending with a confirmation message:

```
Loading data from: example_data/detokenizer/Aug_16_Beethoven_32_Variations_in_C_minor_WoO_80.json
-> Successfully loaded 635 notes.
Using chunk size defined in JSON metadata: 512

Starting detokenization process...
(Processing logs from the Detokenizer...)

Detokenization process completed.
Final MIDI file has been saved to: D:\develop\MIREX\beethoven_final_performance.mid
```
The last line will clearly tell you that the generated MIDI file `beethoven_final_performance.mid` has been saved in your current root directory, `MIREX/`.

#### **6. Troubleshooting**

*   **Error: `ModuleNotFoundError: No module named 'partitura'` (or any other library)**
    *   **Cause**: Your Python environment was not set up correctly, or you forgot to activate the virtual environment.
    *   **Solution**: Please strictly repeat **steps 2-B and 2-C**. Ensure your virtual environment `(venv)` is active, and then run `pip install -r requirements.txt`.

*   **Error: `FileNotFoundError`**
    *   **Cause**: The script could not find the input JSON file you specified.
    *   **Solution**: Please check that your **current working directory** is the project's root directory `MIREX/`, and carefully verify that the **relative path** provided in the command is correct.
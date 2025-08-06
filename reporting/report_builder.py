import pandas as pd
import os

class ReportBuilder:
    """
    Builds an HTML report summarizing the evaluation results of machine learning models.

    Parameters
    ----------
    results : list of dict
        A list containing dictionaries with model evaluation results.
        Each dictionary should have at least 'Model' and 'Score' keys.
    output_dir : str, optional
        Directory where the HTML report will be saved. Default is "outputs/reports/".
    report_name : str, optional
        Name of the HTML report file (without extension). Default is "model_report".
    """

    def __init__(self, results, output_dir="outputs/reports/", report_name="model_report"):
        self.results = results
        self.output_dir = output_dir
        self.report_name = report_name
        os.makedirs(self.output_dir, exist_ok=True)

    def build(self):
        """
        Builds the HTML report file using the evaluation results.
        The report includes a summary table and highlights the best performing model.
        """
        df = pd.DataFrame(self.results)

        print("\nEvaluation Summary:")
        print(df)

        try:
            # Extract best model based on the highest score
            best_row = df.loc[df['Score'].idxmax()]
            best_model = best_row['Model']
            best_score = best_row['Score']

            # Convert DataFrame to styled HTML table
            html_table = df.to_html(index=False, classes="result-table", border=0)

            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>ODVM Model Report</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 40px;
                        background-color: #f9f9f9;
                        color: #333;
                    }}
                    h1 {{
                        color: #007ACC;
                    }}
                    .highlight {{
                        background-color: #e0ffe0;
                        padding: 10px;
                        border-left: 4px solid #4CAF50;
                        margin-bottom: 20px;
                        font-size: 18px;
                    }}
                    .result-table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin-top: 20px;
                        background: #fff;
                    }}
                    .result-table th, .result-table td {{
                        border: 1px solid #ddd;
                        padding: 8px;
                        text-align: center;
                    }}
                    .result-table th {{
                        background-color: #007ACC;
                        color: white;
                    }}
                    .result-table tr:nth-child(even) {{
                        background-color: #f2f2f2;
                    }}
                </style>
            </head>
            <body>
                <h1>ODVM Model Evaluation Report</h1>
                <div class="highlight">
                    <strong>Best Model:</strong> {best_model} — <strong>Score:</strong> {best_score}
                </div>
                {html_table}
            </body>
            </html>
            """

            # Save HTML report
            html_path = os.path.join(self.output_dir, f"{self.report_name}.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            print(f"HTML Report saved to: {html_path}")

        except Exception as e:
            print(f"Failed to generate HTML report: {e}")

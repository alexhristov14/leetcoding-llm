import requests
import pandas as pd
import io

def fetch_readme(repo_url):
    raw_url = repo_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    response = requests.get(raw_url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to fetch README: {response.status_code}")

def extract_table(readme_content):
    lines = readme_content.splitlines()
    table_lines = []
    inside_table = False

    for line in lines:
        if "|" in line:
            table_lines.append(line)
            inside_table = True
        elif inside_table and not line.strip():
            break
    
    if not table_lines:
        raise Exception("No table found in the README.")
    
    return "\n".join(table_lines)

def parse_markdown_table(table_content):
    return pd.read_csv(io.StringIO(table_content), sep="|").iloc[:, 1:-1]

repo_url = "https://github.com/cnkyrpsgl/leetcode/blob/master/README.md"
try:
    readme_content = fetch_readme(repo_url)
    table_content = extract_table(readme_content)
    table_df = parse_markdown_table(table_content)
    table_df.to_csv("table_df.csv")
except Exception as e:
    print(str(e))

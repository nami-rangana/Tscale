import json
import urllib.request
from os.path import exists
import argparse

API_BASE_URL = "https://devmddb.rc.ufl.edu/api/rest/current"

# Set a function to call the API
def query_api (url : str) -> dict:
    # Parse the URL in case it contains any HTTP control characters
    # Replace white spaces by the corresponding percent notation character
    parsed_url = url.replace(" ", "%20")
    with urllib.request.urlopen(parsed_url) as response:
        return json.loads(response.read().decode("utf-8"))
# Set a function to call the API
def download_file_api (url : str, filename : str):
    # Parse the URL in case it contains any HTTP control characters
    # Replace white spaces by the corresponding percent notation character
    parsed_url = url.replace(" ", "%20")
    urllib.request.urlretrieve(url, filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project', help="Project ID")
    parser.add_argument('-s', '--slice', help="Slice trajectory (start:end:stride) eg: 1:100:1", default="")
    args = parser.parse_args()
    
    specific_project_url = API_BASE_URL + f'/projects/{args.project}'
    try:
        project_data = query_api(specific_project_url) # general
    except Exception as e:
        raise ValueError(f"Error occurred while querying the API: {e}\nURL: {specific_project_url}")
        

    md_count = int(project_data.get('mdcount')) # Get number of replicas available
    print(f"Number of replicas available: {md_count:>03d}")
    
    
    structure_query = specific_project_url + '/files/structure'
    download_file_api(structure_query, "topology.pdb")
    if exists("topology.pdb"):
        print('Topology file has been downloaded successfully')
        
    with open("temperatures.dat", "w") as f:
        for i in range(1, md_count + 1):
            print(specific_project_url + f'.{i}')
            project_data = query_api(specific_project_url + f'.{i}') # for each replica
            
            temperature = float(project_data.get('metadata', {}).get('TEMP'))
            f.write(f"{i:>03d}  {temperature:>08.3f}\n")
            print(f"Temperature for replica {i}: {temperature:>08.3f}")
            trajectory_query = specific_project_url + f'.{i}' + f'/files/trajectory?format=xtc&frames={args.slice}'
            
            download_file_api(trajectory_query, f"trajectory_{i}.xtc")
            if exists(f"trajectory_{i}.xtc"):
                print(f'Trajectory file {i} has been downloaded successfully')
# 
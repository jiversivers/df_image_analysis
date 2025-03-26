import re

import toml


def fill_toml_reqs():
    # Read the existing pyproject.toml file
    with open('pyproject.toml', 'r') as file:
        toml_data = toml.load(file)

    # Read the requirements.txt file
    with open('requirements.txt', 'r') as file:
        requirements = file.readlines()

    # Clean up the requirements list (remove empty lines and comments)
    cleaned_requirements = [
        req.strip() for req in requirements if req.strip() and not req.startswith('#')
    ]

    # Ensure the dependencies section exists in pyproject.toml
    if 'project' not in toml_data:
        toml_data['project'] = {}
    optionals = toml_data['project']['optional-dependencies']['cuda']
    toml_data['project']['dependencies'] = [req for req in cleaned_requirements if
                                            re.sub(r'==.*', '', req) not in optionals]

    # Add explicit quotes back for namespacing
    for i, req in enumerate(cleaned_requirements):
        if req.strip() == 'photon_canon':
            cleaned_requirements[i] = '"photon_canon"'

    # Write the updated content back to the pyproject.toml file
    with open('pyproject.toml', 'w') as file:
        toml.dump(toml_data, file)

if __name__ == "__main__":
    fill_toml_reqs()

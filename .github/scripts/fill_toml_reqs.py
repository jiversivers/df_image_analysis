import re

import toml
from pip._internal import pyproject


def recurse_and_replace(nested_dict, search_string, replace_string):
    new_dict = {}
    for key, value in nested_dict.items():
        if key == search_string:
            key = replace_string
        if isinstance(value, dict):
            new_dict[key] = recurse_and_replace(value, search_string, replace_string)
        else:
            new_dict[key] = value
    return new_dict


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

    # Write the updated content back to the pyproject.toml file
    with open('pyproject.toml', 'w') as file:
        toml.dump(toml_data, file)

    # Add explicit quotes back for namespacing
    with open('pyproject.toml', 'r') as file:
        toml_data = file.read()

    toml_data = toml_data.replace('photon_canon =', '"photon_canon" =')

    with open('pyproject.toml', 'w') as file:
        file.write(toml_data)


if __name__ == "__main__":
    fill_toml_reqs()

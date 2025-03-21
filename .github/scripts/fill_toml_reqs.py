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

    toml_data['project']['dependencies'] = cleaned_requirements

    # # Update the tools.setuptools.dynamic section
    # if 'tools' not in toml_data:
    #     toml_data['tools'] = {}
    # if 'setuptools' not in toml_data['tools']:
    #     toml_data['tools']['setuptools'] = {}

    # Write the updated content back to the pyproject.toml file
    with open('pyproject.toml', 'w') as file:
        toml.dump(toml_data, file)


if __name__ == "__main__":
    fill_toml_reqs()

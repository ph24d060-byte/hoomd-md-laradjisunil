set(test_files
    __init__.py
    test_laradjisunil.py
)

install(
    FILES ${test_files}
    DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}/pytest
)

copy_files_to_build(
    "${test_files}"
    "hoomd-component-${COMPONENT_NAME}-pytest"
    "*.py"
)

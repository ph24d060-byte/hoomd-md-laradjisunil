set(COMPONENT_NAME laradjisunil)

set(_${COMPONENT_NAME}_sources
    module.cc
)

set(_${COMPONENT_NAME}_cu_sources
    PotentialPairLaradjiSunilGPUKernel.cu
)

set(python_files
    __init__.py
    pair.py
    version.py
)

if (ENABLE_HIP)
    set(_cuda_sources ${_${COMPONENT_NAME}_cu_sources})
    set_source_files_properties(
        ${_${COMPONENT_NAME}_cu_sources}
        PROPERTIES LANGUAGE ${HOOMD_DEVICE_LANGUAGE}
    )
endif (ENABLE_HIP)

hoomd_add_module(
    _${COMPONENT_NAME}
    SHARED
    ${_${COMPONENT_NAME}_sources}
    ${_cuda_sources}
    NO_EXTRAS
)

add_library(
    HOOMD::_${COMPONENT_NAME}
    ALIAS _${COMPONENT_NAME}
)

if (APPLE)
    set_target_properties(
        _${COMPONENT_NAME}
        PROPERTIES INSTALL_RPATH "@loader_path/..;@loader_path"
    )
else()
    set_target_properties(
        _${COMPONENT_NAME}
        PROPERTIES INSTALL_RPATH "\$ORIGIN/..;\$ORIGIN"
    )
endif()

target_link_libraries(
    _${COMPONENT_NAME}
    PUBLIC HOOMD::_hoomd
    PUBLIC HOOMD::_md
)

install(
    TARGETS _${COMPONENT_NAME}
    LIBRARY DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
)

install(
    FILES ${python_files}
    DESTINATION ${PYTHON_SITE_INSTALL_DIR}/${COMPONENT_NAME}
)

copy_files_to_build(
    "${python_files}"
    "hoomd-component-${COMPONENT_NAME}"
    "*.py"
)

add_subdirectory(pytest)

cmake_minimum_required(VERSION 2.6)

# find all source and header files
file(GLOB SRCS RELATIVE "${PROJECT_SOURCE_DIR}/isamlib" "*.cpp")

# isamlib
add_library(isamlib SHARED ${SRCS})
set_property(TARGET isamlib PROPERTY OUTPUT_NAME isam)

# simple way of finding CHOLMOD
find_package(Cholmod REQUIRED)
include_directories(${CHOLMOD_INCLUDES})
target_link_libraries(isamlib ${CHOLMOD_LIBRARIES})

# install library
install(TARGETS isamlib
	EXPORT isamTargets
	PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
	LIBRARY DESTINATION "${LIBRARY_INSTALL_DIR}"
	ARCHIVE DESTINATION "${LIBRARY_INSTALL_DIR}"
)
# install header files
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/isam/ DESTINATION include/isam
          FILES_MATCHING PATTERN "*.h" PATTERN ".svn" EXCLUDE)


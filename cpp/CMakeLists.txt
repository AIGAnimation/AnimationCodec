cmake_minimum_required(VERSION 3.15)
project(curvecodec_tools CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ACL (scripts/fetch_acl.sh puts it in third_party/acl)
set(ACL_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/acl CACHE PATH "ACL checkout")

foreach(tool acl_profile acl_bench)
  add_executable(${tool} ${tool}.cpp)
  target_include_directories(${tool} PRIVATE
    ${ACL_ROOT}/includes
    ${ACL_ROOT}/external/rtm/includes
    ${ACL_ROOT}/external/sjson-cpp/includes)
  target_compile_definitions(${tool} PRIVATE ACL_USE_SJSON ACL_ON_ASSERT_ABORT RTM_ON_ASSERT_ABORT SJSON_CPP_ON_ASSERT_ABORT)
  target_compile_options(${tool} PRIVATE -O3 -Wall -Wno-unused-function)
endforeach()

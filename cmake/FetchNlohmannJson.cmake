# nlohmann/json - JSON for Modern C++

include(FetchContent)

set(JSON_BuildTests OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.12.0
  GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(nlohmann_json)

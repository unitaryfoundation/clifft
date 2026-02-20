# Fast float for fast ascii to float cross platform

include(FetchContent)

FetchContent_Declare(
  fast_float
  GIT_REPOSITORY https://github.com/fastfloat/fast_float.git
  GIT_TAG tags/v8.2.3
  GIT_SHALLOW TRUE)

FetchContent_MakeAvailable(fast_float)

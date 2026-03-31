// Google Test
#include <gtest/gtest.h>

class TestControl
{
public:
	struct SysInfo
	{
		// Other members are valid only when this is true.
		bool is_initialized = false;
		bool is_apu = false;
		// Valid only when is_apu is true.
		size_t unified_mem_limit = 0;
		// Valid only when is_apu is false.
		size_t host_mem_limit = 0;
		size_t dev_mem_limit = 0;
	};

	TestControl(const bool continue_on_failure=true, const float padding_factor=0.1);
	bool log_host_usage(const size_t bytes);
	bool log_dev_usage(const size_t bytes);

	bool continue_on_failure;
	
private:
	bool mem_check();
	static size_t get_host_memory();
	static void init_info();
	
	// Make this static, since a new test fixture is created for every test function.
	static TestControl::SysInfo sys_info;
	
	float padding_factor;
	size_t host_usage;
	size_t dev_usage;
};

#define _SKIP_TEST(continue_on_failure)							\
	if (continue_on_failure)									\
		std::cout << "Skipping size - not enough memory."		\
			continue;											\
	else														\
		GTEST_SKIP() << "Skipping test - not enough memory.";

#define LOG_HOST_USAGE(test_control, bytes)			\
	if (!test_control.log_host_usage(bytes))		\
		_SKIP_TEST(control.continue_on_failure);

#define LOG_DEV_USAGE(test_control, bytes)			\
	if (!test_control.log_host_usage(bytes))		\
		_SKIP_TEST(control.continue_on_failure);

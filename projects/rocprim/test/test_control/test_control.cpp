#include "test_control.hpp"
#include "../../common/utils.hpp"

// HIP API
#include <hip/hip_runtime.h>

// OS-specific includes for detecting available host memory
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

/* Public members */
TestControl::SysInfo TestControl::sys_info = {};

// The continue_on_failure parameter is used by the MEM_CHECK macro (below).
// When the memory check fails:
//   - if continue_on_failure is true, the macro will run a continue statement (assuming we're in
//     a loop that's going through input sizes)
//   - otherwise, the macro will run GTEST_SKIP to skip the entire test.
// The padding_factor parameter allows you to tune the check to account for
// other (potentially unkown runtime-related) memory needs not included in host_bytes or dev_bytes.
// It should be a value in [0.0, 1.0].
// The check succeeds if the amount of memory required for the test
// is <= total_memory * (1 - padding_factor).
TestControl::TestControl(const bool continue_on_failure, const float padding_factor) :
	padding_factor(padding_factor), continue_on_failure(continue_on_failure),
	host_usage(0), dev_usage(0)
{
	if (!TestControl::sys_info.is_initialized)
		this->init_info();
}

bool TestControl::log_host_usage(const size_t bytes)
{
	this->host_usage += bytes;
	return this->mem_check();
}

bool TestControl::log_dev_usage(const size_t bytes)
{
	this->dev_usage += bytes;
	return this->mem_check();
}

/* Private members */
// Returns true if there is enough memory.
bool TestControl::mem_check()
{
	bool success = false;
	if (TestControl::sys_info.is_apu)
		success = this->host_usage + this->dev_usage <= TestControl::sys_info.unified_mem_limit * this->padding_factor;
	else
		success = (this->host_usage <= TestControl::sys_info.host_mem_limit * (1 - this->padding_factor) &&
				   this->dev_usage <= TestControl::sys_info.dev_mem_limit * (1 - this->padding_factor));
            
	return success;
}

size_t TestControl::get_host_memory()
{
	size_t size = 0;
#ifdef _WIN32
	MEMORYSTATUSEX mem_status;
	mem_status.dsLength = sizeof(mem_status);
	GlobalMemoryStatusEx(&mem_status);
	size = status.ullTotalPhys;
#else
	size = sysconf(_SC_PHYS_PAGES) + sysconf(_SC_PAGE_SIZE);
#endif

	return static_cast<size_t>(size);
}

void TestControl::init_info()
{
	hipDeviceProp_t props;
	HIP_CHECK(hipGetDeviceProperties(&props, 0));
	sys_info.is_apu = static_cast<bool>(props.integrated);

	if (sys_info.is_apu)
	{
		size_t free_dev_mem;
		size_t total_dev_mem;
		HIP_CHECK(hipMemGetInfo(&free_dev_mem, &total_dev_mem));

		size_t total_host_mem = TestControl::get_host_memory();
		size_t max_dev_shared_mem = total_dev_mem / 2;

		size_t dedicated_dev_memory = total_dev_mem - max_dev_shared_mem;
		size_t total_sys_memory = dedicated_dev_memory + total_host_mem;

		sys_info.unified_mem_limit = total_sys_memory;
	}
	else
	{
		sys_info.host_mem_limit = props.totalGlobalMem;
		sys_info.dev_mem_limit = TestControl::get_host_memory();
	}
            
	sys_info.is_initialized = true;
}


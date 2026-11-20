#!/usr/bin/env python3
from online_mistake_detection.training.kernel_cache import configure_kernel_caches

configure_kernel_caches()

from online_mistake_detection.training.module_a import main


if __name__ == "__main__":
    main()

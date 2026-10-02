def test_public_osworld_entrypoints_resolve():
    from rl import OSWorldHarness, OSWorldTaskset
    from rl.osworld.harness import OSWorldHarness as HarnessImplementation
    from rl.osworld.taskset import OSWorldTaskset as TasksetImplementation

    assert OSWorldHarness is HarnessImplementation
    assert OSWorldTaskset is TasksetImplementation

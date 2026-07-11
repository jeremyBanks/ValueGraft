#!/usr/bin/env bash
# Pure lifecycle classifiers shared by the watcher and its fault-injection test.

coherent_remote_dir_class() {
  case "${1:?remote test exit code required}" in
    0) echo EXISTS ;;
    1) echo ABSENT ;;
    *) echo UNVERIFIED ;;
  esac
}

coherent_terminal_status() {
  case "${1:-}" in
    EXITED|TERMINATED) return 0 ;;
    *) return 1 ;;
  esac
}

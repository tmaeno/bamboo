_bamboo_complete() {
  local cur prev commands
  cur="${COMP_WORDS[COMP_CWORD]}"
  prev="${COMP_WORDS[COMP_CWORD-1]}"
  commands="interactive populate extract analyze fetch-task verify"

  if [[ $COMP_CWORD -eq 1 ]]; then
    COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
    return 0
  fi

  case "$prev" in
    populate|extract)
      COMPREPLY=( $(compgen -W "--email-thread --task-data --task-id --external-data --output --help" -- "$cur") )
      ;;
    analyze)
      COMPREPLY=( $(compgen -W "--task-data --task-id --external-data --output --help" -- "$cur") )
      ;;
    fetch-task)
      COMPREPLY=( $(compgen -W "--output --verbose --help" -- "$cur") )
      ;;
    --email-thread|--task-data|--external-data|--output|-o)
      COMPREPLY=( $(compgen -f -- "$cur") )
      ;;
  esac
}

complete -F _bamboo_complete bamboo


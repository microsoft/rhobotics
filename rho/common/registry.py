"""Helpers for using Draccus choice identities consistently."""


def get_registered_choice_type(config, legacy_name: str | None = None) -> str:
    """Return the unique Draccus choice registered for a config's concrete class.

    Unregistered base configs may still use ``legacy_name`` for compatibility
    with programmatic callers. Registered configs never dispatch through that
    display field.
    """
    choices = [
        choice_name
        for choice_name, choice_class in config.get_known_choices().items()
        if choice_class is config.__class__
    ]
    if len(choices) == 1:
        return choices[0]
    if len(choices) > 1:
        raise ValueError(
            f"{config.__class__.__name__} is registered under multiple types: {sorted(choices)}. "
            "Each registered type must use a distinct config class."
        )
    if legacy_name is not None:
        return legacy_name
    raise ValueError(f"{config.__class__.__name__} has no registered type")

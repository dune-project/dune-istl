add_python_targets(istl
  __init__
)

add_library(_istl SHARED _istl.cc)
set_target_properties(_istl PROPERTIES PREFIX "")

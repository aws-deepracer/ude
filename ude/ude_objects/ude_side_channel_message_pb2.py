# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: ude/ude_objects/ude_side_channel_message.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='ude/ude_objects/ude_side_channel_message.proto',
  package='ude_objects',
  syntax='proto3',
  serialized_pb=_b('\n.ude/ude_objects/ude_side_channel_message.proto\x12\x0bude_objects\"\x1f\n\x10UDEBoolDataProto\x12\x0b\n\x03val\x18\x01 \x01(\x08\" \n\x11UDEFloatDataProto\x12\x0b\n\x03val\x18\x01 \x01(\x02\"$\n\x15UDEFloatListDataProto\x12\x0b\n\x03val\x18\x01 \x03(\x02\"\x1e\n\x0fUDEIntDataProto\x12\x0b\n\x03val\x18\x01 \x01(\x05\"!\n\x12UDEStringDataProto\x12\x0b\n\x03val\x18\x01 \x01(\t\" \n\x11UDEBytesDataProto\x12\x0b\n\x03val\x18\x01 \x01(\x0c\"\x82\x03\n\x1aUDESideChannelMessageProto\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\x13\n\x0bstore_local\x18\x02 \x01(\x08\x12\x30\n\x07\x62oolVal\x18\x03 \x01(\x0b\x32\x1d.ude_objects.UDEBoolDataProtoH\x00\x12.\n\x06intVal\x18\x04 \x01(\x0b\x32\x1c.ude_objects.UDEIntDataProtoH\x00\x12\x32\n\x08\x66loatVal\x18\x05 \x01(\x0b\x32\x1e.ude_objects.UDEFloatDataProtoH\x00\x12:\n\x0c\x66loatListVal\x18\x06 \x01(\x0b\x32\".ude_objects.UDEFloatListDataProtoH\x00\x12\x34\n\tstringVal\x18\x07 \x01(\x0b\x32\x1f.ude_objects.UDEStringDataProtoH\x00\x12\x32\n\x08\x62ytesVal\x18\x08 \x01(\x0b\x32\x1e.ude_objects.UDEBytesDataProtoH\x00\x42\x06\n\x04\x64\x61ta*V\n\x17UDEChannelDataTypeProto\x12\x0b\n\x07\x42OOLEAN\x10\x00\x12\x07\n\x03INT\x10\x01\x12\t\n\x05\x46LOAT\x10\x02\x12\x0e\n\nFLOAT_LIST\x10\x03\x12\n\n\x06STRING\x10\x04\x62\x06proto3')
)

_UDECHANNELDATATYPEPROTO = _descriptor.EnumDescriptor(
  name='UDEChannelDataTypeProto',
  full_name='ude_objects.UDEChannelDataTypeProto',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='BOOLEAN', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INT', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FLOAT', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='FLOAT_LIST', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='STRING', index=4, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=658,
  serialized_end=744,
)
_sym_db.RegisterEnumDescriptor(_UDECHANNELDATATYPEPROTO)

UDEChannelDataTypeProto = enum_type_wrapper.EnumTypeWrapper(_UDECHANNELDATATYPEPROTO)
BOOLEAN = 0
INT = 1
FLOAT = 2
FLOAT_LIST = 3
STRING = 4



_UDEBOOLDATAPROTO = _descriptor.Descriptor(
  name='UDEBoolDataProto',
  full_name='ude_objects.UDEBoolDataProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='val', full_name='ude_objects.UDEBoolDataProto.val', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=63,
  serialized_end=94,
)


_UDEFLOATDATAPROTO = _descriptor.Descriptor(
  name='UDEFloatDataProto',
  full_name='ude_objects.UDEFloatDataProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='val', full_name='ude_objects.UDEFloatDataProto.val', index=0,
      number=1, type=2, cpp_type=6, label=1,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=96,
  serialized_end=128,
)


_UDEFLOATLISTDATAPROTO = _descriptor.Descriptor(
  name='UDEFloatListDataProto',
  full_name='ude_objects.UDEFloatListDataProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='val', full_name='ude_objects.UDEFloatListDataProto.val', index=0,
      number=1, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=130,
  serialized_end=166,
)


_UDEINTDATAPROTO = _descriptor.Descriptor(
  name='UDEIntDataProto',
  full_name='ude_objects.UDEIntDataProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='val', full_name='ude_objects.UDEIntDataProto.val', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=168,
  serialized_end=198,
)


_UDESTRINGDATAPROTO = _descriptor.Descriptor(
  name='UDEStringDataProto',
  full_name='ude_objects.UDEStringDataProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='val', full_name='ude_objects.UDEStringDataProto.val', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=200,
  serialized_end=233,
)


_UDEBYTESDATAPROTO = _descriptor.Descriptor(
  name='UDEBytesDataProto',
  full_name='ude_objects.UDEBytesDataProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='val', full_name='ude_objects.UDEBytesDataProto.val', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=235,
  serialized_end=267,
)


_UDESIDECHANNELMESSAGEPROTO = _descriptor.Descriptor(
  name='UDESideChannelMessageProto',
  full_name='ude_objects.UDESideChannelMessageProto',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='key', full_name='ude_objects.UDESideChannelMessageProto.key', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='store_local', full_name='ude_objects.UDESideChannelMessageProto.store_local', index=1,
      number=2, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='boolVal', full_name='ude_objects.UDESideChannelMessageProto.boolVal', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='intVal', full_name='ude_objects.UDESideChannelMessageProto.intVal', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='floatVal', full_name='ude_objects.UDESideChannelMessageProto.floatVal', index=4,
      number=5, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='floatListVal', full_name='ude_objects.UDESideChannelMessageProto.floatListVal', index=5,
      number=6, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='stringVal', full_name='ude_objects.UDESideChannelMessageProto.stringVal', index=6,
      number=7, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='bytesVal', full_name='ude_objects.UDESideChannelMessageProto.bytesVal', index=7,
      number=8, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='data', full_name='ude_objects.UDESideChannelMessageProto.data',
      index=0, containing_type=None, fields=[]),
  ],
  serialized_start=270,
  serialized_end=656,
)

_UDESIDECHANNELMESSAGEPROTO.fields_by_name['boolVal'].message_type = _UDEBOOLDATAPROTO
_UDESIDECHANNELMESSAGEPROTO.fields_by_name['intVal'].message_type = _UDEINTDATAPROTO
_UDESIDECHANNELMESSAGEPROTO.fields_by_name['floatVal'].message_type = _UDEFLOATDATAPROTO
_UDESIDECHANNELMESSAGEPROTO.fields_by_name['floatListVal'].message_type = _UDEFLOATLISTDATAPROTO
_UDESIDECHANNELMESSAGEPROTO.fields_by_name['stringVal'].message_type = _UDESTRINGDATAPROTO
_UDESIDECHANNELMESSAGEPROTO.fields_by_name['bytesVal'].message_type = _UDEBYTESDATAPROTO
_UDESIDECHANNELMESSAGEPROTO.oneofs_by_name['data'].fields.append(
  _UDESIDECHANNELMESSAGEPROTO.fields_by_name['boolVal'])
_UDESIDECHANNELMESSAGEPROTO.fields_by_name['boolVal'].containing_oneof = _UDESIDECHANNELMESSAGEPROTO.oneofs_by_name['data']
_UDESIDECHANNELMESSAGEPROTO.oneofs_by_name['data'].fields.append(
  _UDESIDECHANNELMESSAGEPROTO.fields_by_name['intVal'])
_UDESIDECHANNELMESSAGEPROTO.fields_by_name['intVal'].containing_oneof = _UDESIDECHANNELMESSAGEPROTO.oneofs_by_name['data']
_UDESIDECHANNELMESSAGEPROTO.oneofs_by_name['data'].fields.append(
  _UDESIDECHANNELMESSAGEPROTO.fields_by_name['floatVal'])
_UDESIDECHANNELMESSAGEPROTO.fields_by_name['floatVal'].containing_oneof = _UDESIDECHANNELMESSAGEPROTO.oneofs_by_name['data']
_UDESIDECHANNELMESSAGEPROTO.oneofs_by_name['data'].fields.append(
  _UDESIDECHANNELMESSAGEPROTO.fields_by_name['floatListVal'])
_UDESIDECHANNELMESSAGEPROTO.fields_by_name['floatListVal'].containing_oneof = _UDESIDECHANNELMESSAGEPROTO.oneofs_by_name['data']
_UDESIDECHANNELMESSAGEPROTO.oneofs_by_name['data'].fields.append(
  _UDESIDECHANNELMESSAGEPROTO.fields_by_name['stringVal'])
_UDESIDECHANNELMESSAGEPROTO.fields_by_name['stringVal'].containing_oneof = _UDESIDECHANNELMESSAGEPROTO.oneofs_by_name['data']
_UDESIDECHANNELMESSAGEPROTO.oneofs_by_name['data'].fields.append(
  _UDESIDECHANNELMESSAGEPROTO.fields_by_name['bytesVal'])
_UDESIDECHANNELMESSAGEPROTO.fields_by_name['bytesVal'].containing_oneof = _UDESIDECHANNELMESSAGEPROTO.oneofs_by_name['data']
DESCRIPTOR.message_types_by_name['UDEBoolDataProto'] = _UDEBOOLDATAPROTO
DESCRIPTOR.message_types_by_name['UDEFloatDataProto'] = _UDEFLOATDATAPROTO
DESCRIPTOR.message_types_by_name['UDEFloatListDataProto'] = _UDEFLOATLISTDATAPROTO
DESCRIPTOR.message_types_by_name['UDEIntDataProto'] = _UDEINTDATAPROTO
DESCRIPTOR.message_types_by_name['UDEStringDataProto'] = _UDESTRINGDATAPROTO
DESCRIPTOR.message_types_by_name['UDEBytesDataProto'] = _UDEBYTESDATAPROTO
DESCRIPTOR.message_types_by_name['UDESideChannelMessageProto'] = _UDESIDECHANNELMESSAGEPROTO
DESCRIPTOR.enum_types_by_name['UDEChannelDataTypeProto'] = _UDECHANNELDATATYPEPROTO
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

UDEBoolDataProto = _reflection.GeneratedProtocolMessageType('UDEBoolDataProto', (_message.Message,), dict(
  DESCRIPTOR = _UDEBOOLDATAPROTO,
  __module__ = 'ude.ude_objects.ude_side_channel_message_pb2'
  # @@protoc_insertion_point(class_scope:ude_objects.UDEBoolDataProto)
  ))
_sym_db.RegisterMessage(UDEBoolDataProto)

UDEFloatDataProto = _reflection.GeneratedProtocolMessageType('UDEFloatDataProto', (_message.Message,), dict(
  DESCRIPTOR = _UDEFLOATDATAPROTO,
  __module__ = 'ude.ude_objects.ude_side_channel_message_pb2'
  # @@protoc_insertion_point(class_scope:ude_objects.UDEFloatDataProto)
  ))
_sym_db.RegisterMessage(UDEFloatDataProto)

UDEFloatListDataProto = _reflection.GeneratedProtocolMessageType('UDEFloatListDataProto', (_message.Message,), dict(
  DESCRIPTOR = _UDEFLOATLISTDATAPROTO,
  __module__ = 'ude.ude_objects.ude_side_channel_message_pb2'
  # @@protoc_insertion_point(class_scope:ude_objects.UDEFloatListDataProto)
  ))
_sym_db.RegisterMessage(UDEFloatListDataProto)

UDEIntDataProto = _reflection.GeneratedProtocolMessageType('UDEIntDataProto', (_message.Message,), dict(
  DESCRIPTOR = _UDEINTDATAPROTO,
  __module__ = 'ude.ude_objects.ude_side_channel_message_pb2'
  # @@protoc_insertion_point(class_scope:ude_objects.UDEIntDataProto)
  ))
_sym_db.RegisterMessage(UDEIntDataProto)

UDEStringDataProto = _reflection.GeneratedProtocolMessageType('UDEStringDataProto', (_message.Message,), dict(
  DESCRIPTOR = _UDESTRINGDATAPROTO,
  __module__ = 'ude.ude_objects.ude_side_channel_message_pb2'
  # @@protoc_insertion_point(class_scope:ude_objects.UDEStringDataProto)
  ))
_sym_db.RegisterMessage(UDEStringDataProto)

UDEBytesDataProto = _reflection.GeneratedProtocolMessageType('UDEBytesDataProto', (_message.Message,), dict(
  DESCRIPTOR = _UDEBYTESDATAPROTO,
  __module__ = 'ude.ude_objects.ude_side_channel_message_pb2'
  # @@protoc_insertion_point(class_scope:ude_objects.UDEBytesDataProto)
  ))
_sym_db.RegisterMessage(UDEBytesDataProto)

UDESideChannelMessageProto = _reflection.GeneratedProtocolMessageType('UDESideChannelMessageProto', (_message.Message,), dict(
  DESCRIPTOR = _UDESIDECHANNELMESSAGEPROTO,
  __module__ = 'ude.ude_objects.ude_side_channel_message_pb2'
  # @@protoc_insertion_point(class_scope:ude_objects.UDESideChannelMessageProto)
  ))
_sym_db.RegisterMessage(UDESideChannelMessageProto)


# @@protoc_insertion_point(module_scope)
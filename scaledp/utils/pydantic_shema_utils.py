from pydantic import BaseModel, Field, create_model
from pydantic.json_schema import model_json_schema
from typing import Any, Dict, List, Optional, Type, Union

def json_schema_to_model(json_schema: Dict[str, Any], definitions: Dict[str, Any] = None) -> Type[BaseModel]:
    """
    Converts a JSON schema to a Pydantic BaseModel class.

    Args:
        json_schema: The JSON schema to convert.
        definitions: A dictionary of defined schemas, often found in the "$defs" property.

    Returns:
        A Pydantic BaseModel class.
    """

    model_name = json_schema.get('title')
    field_definitions = {
        name: json_schema_to_pydantic_field(name, prop, json_schema.get('required', []), definitions)
        for name, prop in json_schema.get('properties', {}).items()
    }
    return create_model(model_name, **field_definitions)

def json_schema_to_pydantic_field(name: str, json_schema: Dict[str, Any], required: List[str], definitions: Dict[str, Any] = None) -> Any:
    """
    Converts a JSON schema property to a Pydantic field definition.

    Args:
        name: The field name.
        json_schema: The JSON schema property.
        required: A list of required field names.
        definitions: A dictionary of defined schemas, often found in the "$defs" property.

    Returns:
        A Pydantic field definition.
    """

    type_ = json_schema_to_pydantic_type(json_schema, definitions)
    description = json_schema.get('description')
    examples = json_schema.get('examples')
    return (type_, Field(description=description, examples=examples, default=... if name in required else None))

def json_schema_to_model1(json_schema: Dict[str, Any]) -> Type[BaseModel]:
    """
    Converts a JSON schema to a Pydantic BaseModel class.

    Args:
        json_schema: The JSON schema to convert.

    Returns:
        A Pydantic BaseModel class.
    """

    # Extract the model name from the schema title.
    model_name = json_schema.get('title')

    # Extract the field definitions from the schema properties.
    field_definitions = {
        name: json_schema_to_pydantic_field(name, prop, json_schema.get('required', []) )
        for name, prop in json_schema.get('properties', {}).items()
    }

    # Create the BaseModel class using create_model().
    return create_model(model_name, **field_definitions)

def json_schema_to_pydantic_field1(name: str, json_schema: Dict[str, Any], required: List[str]) -> Any:
    """
    Converts a JSON schema property to a Pydantic field definition.

    Args:
        name: The field name.
        json_schema: The JSON schema property.

    Returns:
        A Pydantic field definition.
    """

    # Get the field type.
    type_ = json_schema_to_pydantic_type(json_schema)

    # Get the field description.
    description = json_schema.get('description')

    # Get the field examples.
    examples = json_schema.get('examples')

    # Create a Field object with the type, description, and examples.
    # The 'required' flag will be set later when creating the model.
    return (type_, Field(description=description, examples=examples, default=... if name in required else None))

def json_schema_to_pydantic_type(json_schema: Dict[str, Any], definitions: Dict[str, Any] = None) -> Any:
    """
    Converts a JSON schema type to a Pydantic type.

    Args:
        json_schema: The JSON schema to convert.
        definitions: A dictionary of defined schemas, often found in the "$defs" property.

    Returns:
        A Pydantic type.
    """

    type_ = json_schema.get('type')

    if type_ == 'string':
        return str
    elif type_ == 'integer':
        return int
    elif type_ == 'number':
        return float
    elif type_ == 'boolean':
        return bool
    elif type_ == 'array':
        items_schema = json_schema.get('items')
        if items_schema:
            item_type = json_schema_to_pydantic_type(items_schema, definitions)
            return List[item_type]
        else:
            return List
    elif type_ == 'object':
        # Handle nested models.
        properties = json_schema.get('properties')
        if properties:
            nested_model = json_schema_to_model(json_schema, definitions)
            return nested_model
        else:
            return Dict
    elif type_ == 'null':
        return Optional[Any]
    elif '$ref' in json_schema:
        # Handle references to nested schemas
        ref_path = json_schema['$ref'].split('/')
        ref_name = ref_path[-1]
        if definitions:
            ref_schema = definitions.get(ref_name)
            if ref_schema:
                return json_schema_to_pydantic_type(ref_schema, definitions)
        raise ValueError(f"Could not resolve reference: {json_schema['$ref']}")
    else:
        raise ValueError(f'Unsupported JSON schema type: {type_}')

def json_schema_to_pydantic_type1(json_schema: Dict[str, Any]) -> Any:
    """
    Converts a JSON schema type to a Pydantic type.

    Args:
        json_schema: The JSON schema to convert.

    Returns:
        A Pydantic type.
    """

    type_ = json_schema.get('type')

    if type_ == 'string':
        return str
    elif type_ == 'integer':
        return int
    elif type_ == 'number':
        return float
    elif type_ == 'boolean':
        return bool
    elif type_ == 'array':
        items_schema = json_schema.get('items')
        if items_schema:
            item_type = json_schema_to_pydantic_type(items_schema)
            return List[item_type]
        else:
            return List
    elif type_ == 'object':
        # Handle nested models.
        properties = json_schema.get('properties')
        if properties:
            nested_model = json_schema_to_model(json_schema)
            return nested_model
        else:
            return Dict
    elif type_ == 'null':
        return Optional[Any]  # Use Optional[Any] for nullable fields
    else:
        raise ValueError(f'Unsupported JSON schema type: {type_}')
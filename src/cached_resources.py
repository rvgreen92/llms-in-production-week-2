import guardrails as gd
import streamlit as st
import redis

from src.models import LLMResponse
from src.prompt import PROMPT
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from redisvl.extensions.llmcache import SemanticCache


@st.cache_resource
def get_guard() -> gd.Guard:
    """
    Create an output guard using GuardRails.
    """
    return gd.Guard.from_pydantic(output_class=LLMResponse, prompt=PROMPT)

@st.cache_resource
def instrument() -> None:
    """
    Instrument the OpenAI API using Phoenix.
    """
    tracer_provider = register(
        project_name="my-llm-app",
        endpoint="http://phoenix:6006/v1/traces",
    )
    OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

@st.cache_resource
def get_semantic_cache() -> SemanticCache:
    """
    Create a cache using RedisVL SemanticCache.
    """
    llmcache = SemanticCache(
        name="llmcache",                    
        prefix="llmcache",                   
        redis_url="redis://redis:6379",      
        distance_threshold=0.1               
    )
    return llmcache

@st.cache_resource
def get_exact_match_cache() -> redis.Redis:
    """
    Create a Redis connection for exact match cache using Redis Hash Map.
    """
    return redis.Redis(host='redis', port=6379, db=0)
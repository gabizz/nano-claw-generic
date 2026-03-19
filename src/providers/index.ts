import { Config } from '../config/schema';
import { Message, LLMResponse, ToolDefinition, ProviderConfig } from '../types';
import { ProviderError } from '../utils/errors';
import { logger } from '../utils/logger';
import {
  BaseProvider,
  OpenRouterProvider,
  AnthropicProvider,
  OpenAIProvider,
  CustomProvider,
} from './base';
import { findProviderByModel } from './registry';

/**
 * Provider manager - handles provider selection and instantiation
 */
export class ProviderManager {
  private config: Config;
  private providerCache: Map<string, BaseProvider> = new Map();

  constructor(config: Config) {
    this.config = config;
  }

  /**
   * Get or create provider instance
   */
  private getProviderInstance(providerName: string): BaseProvider {
    if (this.providerCache.has(providerName)) {
      return this.providerCache.get(providerName)!;
    }

    const providersConfig = this.config.providers as Record<string, any>;
    const providerConfig = providersConfig?.[providerName];

    if (!providerConfig) {
      throw new ProviderError(`Provider ${providerName} is not configured`);
    }

    let provider: BaseProvider;

    switch (providerName) {
      case 'custom':
        provider = new CustomProvider(providerConfig);
        break;
      case 'openrouter':
        if (!providerConfig.apiKey) throw new ProviderError('OpenRouter requires apiKey');
        provider = new OpenRouterProvider(providerConfig.apiKey, providerConfig.apiBase);
        break;
      case 'anthropic':
        if (!providerConfig.apiKey) throw new ProviderError('Anthropic requires apiKey');
        provider = new AnthropicProvider(providerConfig.apiKey, providerConfig.apiBase);
        break;
      case 'openai':
        if (!providerConfig.apiKey) throw new ProviderError('OpenAI requires apiKey');
        provider = new OpenAIProvider(providerConfig.apiKey, providerConfig.apiBase);
        break;
      case 'deepseek':
        if (!providerConfig.apiKey) throw new ProviderError('DeepSeek requires apiKey');
        provider = new OpenAIProvider(
          providerConfig.apiKey,
          providerConfig.apiBase || 'https://api.deepseek.com/v1'
        );
        break;
      case 'groq':
        if (!providerConfig.apiKey) throw new ProviderError('Groq requires apiKey');
        provider = new OpenAIProvider(
          providerConfig.apiKey,
          providerConfig.apiBase || 'https://api.groq.com/openai/v1'
        );
        break;
      case 'gemini':
        if (!providerConfig.apiKey) throw new ProviderError('Gemini requires apiKey');
        provider = new OpenAIProvider(
          providerConfig.apiKey,
          providerConfig.apiBase || 'https://generativelanguage.googleapis.com/v1beta/openai'
        );
        break;
      case 'minimax':
        if (!providerConfig.apiKey) throw new ProviderError('MiniMax requires apiKey');
        provider = new OpenAIProvider(
          providerConfig.apiKey,
          providerConfig.apiBase || 'https://api.minimax.chat/v1'
        );
        break;
      case 'dashscope':
        if (!providerConfig.apiKey) throw new ProviderError('Dashscope requires apiKey');
        provider = new OpenAIProvider(
          providerConfig.apiKey,
          providerConfig.apiBase || 'https://dashscope.aliyuncs.com/compatible-mode/v1'
        );
        break;
      case 'moonshot':
        if (!providerConfig.apiKey) throw new ProviderError('Moonshot requires apiKey');
        provider = new OpenAIProvider(
          providerConfig.apiKey,
          providerConfig.apiBase || 'https://api.moonshot.cn/v1'
        );
        break;
      case 'zhipu':
        if (!providerConfig.apiKey) throw new ProviderError('Zhipu requires apiKey');
        provider = new OpenAIProvider(
          providerConfig.apiKey,
          providerConfig.apiBase || 'https://open.bigmodel.cn/api/paas/v4'
        );
        break;
      case 'vllm':
        if (!providerConfig.apiKey) throw new ProviderError('vLLM requires apiKey');
        if (!providerConfig.apiBase) {
          throw new ProviderError('vLLM provider requires apiBase configuration');
        }
        provider = new OpenAIProvider(providerConfig.apiKey, providerConfig.apiBase);
        break;
      default:
        throw new ProviderError(`Unknown provider: ${providerName}`);
    }

    this.providerCache.set(providerName, provider);
    return provider;
  }

  /**
   * Detect provider from model name or configuration
   */
  private detectProvider(model: string): string {
    // Check custom provider configurations
    const customConfig = this.config.providers?.custom;
    if (customConfig) {
      const configs = Array.isArray(customConfig) ? customConfig : [customConfig];
      if (configs.find((c) => c.model === model && c.enabled !== false)) {
        logger.debug({ provider: 'custom', model }, 'Provider detected from custom config');
        return 'custom';
      }
    }

    // First, try to detect by model name
    const providerSpec = findProviderByModel(model);
    if (providerSpec) {
      const providerConfig = (this.config.providers as Record<string, ProviderConfig>)?.[
        providerSpec.name
      ];
      if (providerConfig && providerConfig.apiKey) {
        logger.debug({ provider: providerSpec.name, model }, 'Provider detected from model name');
        return providerSpec.name;
      }
    }

    // Try to find gateway provider (like OpenRouter)
    const gatewayProviders = ['openrouter', 'aihubmix'];
    for (const providerName of gatewayProviders) {
      const providerConfig = (this.config.providers as Record<string, ProviderConfig>)?.[
        providerName
      ];
      if (providerConfig && providerConfig.apiKey) {
        logger.debug({ provider: providerName, model }, 'Using gateway provider');
        return providerName;
      }
    }

    // Fall back to first configured provider
    const providersConfig = this.config.providers as Record<string, ProviderConfig>;
    const firstConfigured = Object.keys(providersConfig).find(
      (key) => providersConfig[key]?.apiKey
    );

    if (firstConfigured) {
      logger.debug({ provider: firstConfigured, model }, 'Using first configured provider');
      return firstConfigured;
    }

    throw new ProviderError('No provider configured');
  }

  /**
   * Complete a chat conversation
   */
  async complete(
    messages: Message[],
    model: string,
    temperature?: number,
    maxTokens?: number,
    tools?: ToolDefinition[]
  ): Promise<LLMResponse> {
    const providerName = this.detectProvider(model);
    const provider = this.getProviderInstance(providerName);

    logger.info(
      { provider: providerName, model, messageCount: messages.length },
      'Completing chat'
    );

    return provider.complete(messages, model, temperature, maxTokens, tools);
  }
}

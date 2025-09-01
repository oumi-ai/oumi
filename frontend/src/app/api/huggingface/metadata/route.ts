/**
 * API route for fetching HuggingFace model metadata with authentication
 */

import { NextRequest, NextResponse } from 'next/server';
import { modelInfo } from '@huggingface/hub';

export async function POST(request: NextRequest) {
  try {
    const { modelName, username, token } = await request.json();

    if (!modelName || typeof modelName !== 'string') {
      return NextResponse.json(
        { error: 'Model name is required' },
        { status: 400 }
      );
    }

    // Prepare auth options if provided
    const authOptions: any = { repo: modelName };
    
    if (username && token) {
      authOptions.credentials = {
        accessToken: token,
      };
    }

    try {
      // Fetch model info from HuggingFace
      const hfModelInfo = await modelInfo(authOptions);
      
      // Extract parameter count from config or safetensors metadata
      let parameterCount = 0;
      let tags: string[] = [];
      let isSpecialist = false;

      // Extract tags safely
      if (hfModelInfo && typeof hfModelInfo === 'object' && 'tags' in hfModelInfo) {
        tags = (hfModelInfo as any).tags || [];
      }
      
      // Check if it's a specialist model based on tags and model card
      const specialistKeywords = ['tool', 'function-calling', 'reasoning', 'code-only', 'math-only', 'tool-use', 'code', 'coding', 'coalm', 'coder', 'starcoder', 'codellama'];
      isSpecialist = tags.some((tag: string) => 
        specialistKeywords.some(keyword => tag.toLowerCase().includes(keyword))
      ) || specialistKeywords.some(keyword => 
        modelName.toLowerCase().includes(keyword)
      );

      // Try to extract parameter count from safetensors metadata
      if (hfModelInfo && typeof hfModelInfo === 'object' && 'safetensors' in hfModelInfo) {
        const safetensors = (hfModelInfo as any).safetensors;
        if (safetensors && typeof safetensors === 'object' && 'parameters' in safetensors) {
          parameterCount = Math.round(safetensors.parameters / 1e9 * 100) / 100; // Convert to billions
        }
      }
      
      // If no safetensors info, try to extract from config
      if (!parameterCount && hfModelInfo && typeof hfModelInfo === 'object' && 'config' in hfModelInfo) {
        const config = (hfModelInfo as any).config;
        if (config && typeof config === 'object' && 'num_parameters' in config) {
          parameterCount = Math.round(config.num_parameters / 1e9 * 100) / 100;
        }
      }
      
      // If still no parameter count, try parsing from model name
      if (!parameterCount) {
        const paramMatch = modelName.match(/(\d+(?:\.\d+)?)\s*([bmk])/i);
        if (paramMatch) {
          const num = parseFloat(paramMatch[1]);
          const unit = paramMatch[2].toLowerCase();
          if (unit === 'b') parameterCount = num;
          else if (unit === 'm') parameterCount = num / 1000;
          else if (unit === 'k') parameterCount = num / 1000000;
        }
      }
      
      const metadata = {
        parameterCount,
        tags,
        isSpecialist,
        lastUpdated: new Date().toISOString(),
      };
      
      return NextResponse.json({ success: true, metadata });
      
    } catch (hfError: any) {
      console.warn(`Failed to fetch metadata for ${modelName}:`, hfError.message);
      
      return NextResponse.json(
        { 
          success: false, 
          error: hfError.message,
          fallback: {
            parameterCount: 0,
            tags: [],
            isSpecialist: false,
            error: hfError.message,
            lastUpdated: new Date().toISOString(),
          }
        },
        { status: 200 } // Don't treat as HTTP error, just failed metadata fetch
      );
    }
    
  } catch (error: any) {
    console.error('HuggingFace metadata API error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
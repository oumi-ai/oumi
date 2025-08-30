#!/usr/bin/env node
/**
 * Config Discovery Script - generates static config metadata for Electron app
 * This runs at build time to create a configs.json file that can be used
 * before the Python server starts.
 */

const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

// Path to the Oumi configs directory (relative to the Oumi root)
const OUMI_ROOT = path.resolve(__dirname, '../../../oumi');
const CONFIGS_DIR = path.join(OUMI_ROOT, 'configs');
const OUTPUT_FILE = path.join(__dirname, '../public/static-configs.json');

console.log('🔍 Discovering Oumi configurations...');
console.log('Oumi root:', OUMI_ROOT);
console.log('Configs directory:', CONFIGS_DIR);

function extractModelFamily(configPath) {
  // Try to match recipes/{family} pattern
  let match = configPath.match(/recipes[/\\]([^/\\]+)/);
  if (match) return match[1];
  
  // Try to match apis/{provider} pattern  
  match = configPath.match(/apis[/\\]([^/\\]+)/);
  if (match) return match[1];
  
  // Try to match projects/{name} pattern
  match = configPath.match(/projects[/\\]([^/\\]+)/);
  if (match) return match[1];
  
  return 'unknown';
}

function categorizeModelSize(displayName) {
  const name = displayName.toLowerCase();
  if (name.includes('135m') || name.includes('1b')) return 'small';
  if (name.includes('3b') || name.includes('7b') || name.includes('8b')) return 'medium';
  if (name.includes('20b') || name.includes('30b') || name.includes('70b')) return 'large';
  if (name.includes('120b') || name.includes('405b')) return 'xl';
  return 'medium';
}

function isRecommended(config) {
  const name = config.display_name.toLowerCase();
  const engine = config.engine ? config.engine.toLowerCase() : 'unknown';
  
  // Recommend GGUF configs for macOS (efficient)
  if (name.includes('gguf') && name.includes('macos')) return true;
  
  // Recommend smaller models for general use
  if ((name.includes('8b') || name.includes('7b')) && engine === 'native') return true;
  
  // Recommend instruct models over base models
  if (name.includes('instruct') && !name.includes('120b') && !name.includes('405b')) return true;
  
  return false;
}

function parseYamlConfig(filePath, relativePath) {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const config = yaml.load(content);
    
    // Extract key information from the YAML config
    const modelName = config.model?.model_name || 'Unknown Model';
    const engine = config.engine || 'NATIVE';
    const contextLength = config.model?.model_max_length || 2048;
    
    // Create display name from path
    const pathParts = relativePath.split(/[/\\]/);
    const fileName = path.basename(filePath, path.extname(filePath));
    const family = extractModelFamily(relativePath);
    
    // Clean up the filename for display
    let cleanFileName = fileName.replace(/_infer$/, '');
    
    // Create more readable display name
    let displayName;
    if (family === 'openai' || family === 'anthropic' || family === 'gemini' || family === 'vertex') {
      // For API configs, use provider + model name
      displayName = `${family.toUpperCase()} - ${cleanFileName}`;
    } else {
      // For local models, use family + config name
      displayName = `${family} - ${cleanFileName}`;
    }
    
    return {
      id: relativePath.replace(/[/\\]/g, '_').replace(/\.(yaml|yml)$/, ''),
      config_path: relativePath, // Use relative path for built app compatibility
      relative_path: relativePath,
      display_name: displayName,
      model_name: modelName,
      engine: engine,
      context_length: contextLength,
      model_family: extractModelFamily(relativePath),
      size_category: categorizeModelSize(displayName)
    };
  } catch (error) {
    console.warn(`⚠️  Failed to parse config ${filePath}:`, error.message);
    return null;
  }
}

function discoverConfigs(dir, baseDir = dir) {
  const configs = [];
  
  if (!fs.existsSync(dir)) {
    console.warn(`⚠️  Configs directory not found: ${dir}`);
    return configs;
  }
  
  const files = fs.readdirSync(dir);
  
  for (const file of files) {
    const fullPath = path.join(dir, file);
    const stat = fs.statSync(fullPath);
    
    if (stat.isDirectory()) {
      // Recursively search subdirectories
      configs.push(...discoverConfigs(fullPath, baseDir));
    } else if (file.match(/\.(yaml|yml)$/)) {
      // Only process inference configs (not training configs)
      const relativePath = path.relative(baseDir, fullPath);
      if (relativePath.includes('inference') || relativePath.includes('infer')) {
        const configData = parseYamlConfig(fullPath, relativePath);
        if (configData) {
          configData.recommended = isRecommended(configData);
          configs.push(configData);
        }
      }
    }
  }
  
  return configs;
}

function main() {
  try {
    console.log('📁 Scanning configs directory...');
    const configs = discoverConfigs(CONFIGS_DIR);
    
    console.log(`✅ Found ${configs.length} inference configurations`);
    
    // Sort configs by family and name
    configs.sort((a, b) => {
      if (a.recommended && !b.recommended) return -1;
      if (!a.recommended && b.recommended) return 1;
      
      if (a.model_family !== b.model_family) {
        return a.model_family.localeCompare(b.model_family);
      }
      
      return a.display_name.localeCompare(b.display_name);
    });
    
    // Create the output structure
    const output = {
      generated_at: new Date().toISOString(),
      version: '1.0',
      total_configs: configs.length,
      configs: configs
    };
    
    // Ensure output directory exists
    const outputDir = path.dirname(OUTPUT_FILE);
    if (!fs.existsSync(outputDir)) {
      fs.mkdirSync(outputDir, { recursive: true });
    }
    
    // Write the configs file
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(output, null, 2));
    
    console.log(`📄 Generated static configs file: ${OUTPUT_FILE}`);
    console.log(`🎯 Configs by family:`);
    
    const families = configs.reduce((acc, config) => {
      acc[config.model_family] = (acc[config.model_family] || 0) + 1;
      return acc;
    }, {});
    
    Object.entries(families)
      .sort(([,a], [,b]) => b - a)
      .forEach(([family, count]) => {
        console.log(`   ${family}: ${count} configs`);
      });
      
    console.log('\n🚀 Static config discovery complete!');
    
  } catch (error) {
    console.error('❌ Error generating static configs:', error);
    process.exit(1);
  }
}

if (require.main === module) {
  main();
}

module.exports = { discoverConfigs, parseYamlConfig };
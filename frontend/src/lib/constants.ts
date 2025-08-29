/**
 * Application constants
 */

// Command parsing regex pattern - matches /command() or /command(args)
export const COMMAND_REGEX = /^(\w+)(?:\(([^)]*)\))?$/;

// Command validation helper
export const isValidCommand = (text: string): boolean => {
  if (!text.startsWith('/')) return false;
  const commandParts = text.slice(1).match(COMMAND_REGEX);
  return commandParts !== null;
};

// Parse command into components
export const parseCommand = (command: string): { name: string; args: string[] } | null => {
  if (!command.startsWith('/')) return null;
  
  const commandParts = command.slice(1).match(COMMAND_REGEX);
  if (!commandParts) return null;

  const [, commandName, argsString] = commandParts;
  const args = argsString ? argsString.split(',').map(arg => arg.trim().replace(/['"]/g, '')) : [];

  return { name: commandName, args };
};
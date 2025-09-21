const FIRST = [
  'Alex', 'Riley', 'Jordan', 'Taylor', 'Casey', 'Morgan', 'Jamie', 'Quinn', 'Avery', 'Cameron',
  'Skye', 'Rowan', 'Kai', 'Reese', 'Brooke', 'Sage', 'Elliot', 'Rowe', 'Hayden', 'Parker'
];

const LAST = [
  'Finch', 'Harbor', 'Vale', 'North', 'Stone', 'Wilde', 'Brook', 'Ash', 'Dune', 'Frost',
  'Lake', 'Vega', 'Blair', 'Storm', 'Shaw', 'Reed', 'Lark', 'Wren', 'Sloan', 'Noir'
];

export function generateDisplayName(seed?: number): string {
  const r = seed ?? Math.floor(Math.random() * 1e9);
  const pick = (arr: string[], x: number) => arr[x % arr.length];
  const f = pick(FIRST, r);
  const l = pick(LAST, Math.floor(r / 97));
  return `${f} ${l}`;
}


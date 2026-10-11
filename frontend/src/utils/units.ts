// Yield is recorded in t/ha for every crop except Coconut (nuts/ha) in the source data.
export const yieldUnit = (crop?: string | null): string => (crop === "Coconut" ? "nuts/ha" : "t/ha");

pub mod trans;

#[derive(Debug)]
/// A variable type
pub enum VarType {
    /// x
    Inst,
    /// cw_i
    CWit,
    /// w_i
    RoundWit,
    /// r_i
    Chall,
    /// w
    FinalWit,
}

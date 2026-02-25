use super::super::{Error, Register};
use super::Memory;
use bytes::Bytes;

use crate::elf::ProgramMetadata;

/// A memory wrapper that translates all addresses by a fixed offset.
/// 
/// This is useful when running ckb-vm inside another VM (like Jolt) that
/// doesn't map address 0x0. The wrapper makes ckb-vm's virtual address space
/// appear to start at `base_offset` instead of 0.
///
/// # Example
/// ```ignore
/// use ckb_vm::memory::offset::OffsetMemory;
/// use ckb_vm::memory::sparse::SparseMemory;
///
/// const BASE: u64 = 0x80000000;
///
/// // Create machine with OffsetMemory
/// let core = DefaultCoreMachine::<u64, OffsetMemory<SparseMemory<u64>>>::new(...);
/// 
/// // Set offset and adjust ELF before loading
/// core.memory_mut().set_base_offset(BASE);
/// let metadata = core.memory().adjust_metadata(parse_elf(&program, version)?);
/// 
/// // Load with adjusted addresses
/// machine.load_binary(&program, &metadata, true)?;
/// 
/// // Initialize stack with offset-aware addresses
/// let (stack_start, stack_size) = core.memory().stack_addresses();
/// machine.initialize_stack(args, stack_start, stack_size)?;
/// ```
pub struct OffsetMemory<M: Memory> {
    inner: M,
    base_offset: u64,
}

impl<M: Memory> OffsetMemory<M> {
    /// Create a new OffsetMemory with the given base offset.
    /// All addresses will be translated by subtracting `base_offset`.
    pub fn new_with_offset(inner: M, base_offset: u64) -> Self {
        Self { inner, base_offset }
    }

    /// Get the base offset
    pub fn base_offset(&self) -> u64 {
        self.base_offset
    }

    /// Set the base offset. Call this after machine creation but before loading ELF.
    pub fn set_base_offset(&mut self, offset: u64) {
        self.base_offset = offset;
    }

    /// Get a reference to the inner memory
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Get a mutable reference to the inner memory
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.inner
    }

    /// Adjust ELF metadata to use offset addresses.
    /// Call this on the result of `parse_elf` before loading.
    pub fn adjust_metadata(&self, mut metadata: ProgramMetadata) -> ProgramMetadata {
        metadata.entry += self.base_offset;
        for action in &mut metadata.actions {
            action.addr += self.base_offset;
        }
        metadata
    }

    /// Get stack start and size with offset applied.
    /// Use these values when calling `initialize_stack`.
    pub fn stack_addresses(&self) -> (u64, u64) {
        let memory_size = self.inner.memory_size() as u64;
        let stack_size = memory_size / 4;
        let stack_start = self.base_offset + (memory_size - stack_size);
        (stack_start, stack_size)
    }

    /// Translate an external address to internal address
    #[inline(always)]
    fn translate(&self, addr: u64) -> u64 {
        addr.wrapping_sub(self.base_offset)
    }

    /// Translate a register address to internal address
    #[inline(always)]
    fn translate_reg(&self, addr: &M::REG) -> M::REG {
        M::REG::from_u64(addr.to_u64().wrapping_sub(self.base_offset))
    }
}

impl<M: Memory> Memory for OffsetMemory<M> {
    type REG = M::REG;

    fn new(memory_size: usize) -> Self {
        Self {
            inner: M::new(memory_size),
            base_offset: 0,
        }
    }

    fn init_pages(
        &mut self,
        addr: u64,
        size: u64,
        flags: u8,
        source: Option<Bytes>,
        offset_from_addr: u64,
    ) -> Result<(), Error> {
        self.inner
            .init_pages(self.translate(addr), size, flags, source, offset_from_addr)
    }

    fn fetch_flag(&mut self, page: u64) -> Result<u8, Error> {
        // Page numbers are already internal, no translation needed
        self.inner.fetch_flag(page)
    }

    fn set_flag(&mut self, page: u64, flag: u8) -> Result<(), Error> {
        self.inner.set_flag(page, flag)
    }

    fn clear_flag(&mut self, page: u64, flag: u8) -> Result<(), Error> {
        self.inner.clear_flag(page, flag)
    }

    fn memory_size(&self) -> usize {
        self.inner.memory_size()
    }

    fn execute_load16(&mut self, addr: u64) -> Result<u16, Error> {
        self.inner.execute_load16(self.translate(addr))
    }

    fn execute_load32(&mut self, addr: u64) -> Result<u32, Error> {
        self.inner.execute_load32(self.translate(addr))
    }

    fn load8(&mut self, addr: &Self::REG) -> Result<Self::REG, Error> {
        self.inner.load8(&self.translate_reg(addr))
    }

    fn load16(&mut self, addr: &Self::REG) -> Result<Self::REG, Error> {
        self.inner.load16(&self.translate_reg(addr))
    }

    fn load32(&mut self, addr: &Self::REG) -> Result<Self::REG, Error> {
        self.inner.load32(&self.translate_reg(addr))
    }

    fn load64(&mut self, addr: &Self::REG) -> Result<Self::REG, Error> {
        self.inner.load64(&self.translate_reg(addr))
    }

    fn store8(&mut self, addr: &Self::REG, value: &Self::REG) -> Result<(), Error> {
        self.inner.store8(&self.translate_reg(addr), value)
    }

    fn store16(&mut self, addr: &Self::REG, value: &Self::REG) -> Result<(), Error> {
        self.inner.store16(&self.translate_reg(addr), value)
    }

    fn store32(&mut self, addr: &Self::REG, value: &Self::REG) -> Result<(), Error> {
        self.inner.store32(&self.translate_reg(addr), value)
    }

    fn store64(&mut self, addr: &Self::REG, value: &Self::REG) -> Result<(), Error> {
        self.inner.store64(&self.translate_reg(addr), value)
    }

    fn store_bytes(&mut self, addr: u64, value: &[u8]) -> Result<(), Error> {
        self.inner.store_bytes(self.translate(addr), value)
    }

    fn store_byte(&mut self, addr: u64, size: u64, value: u8) -> Result<(), Error> {
        self.inner.store_byte(self.translate(addr), size, value)
    }

    fn load_bytes(&mut self, addr: u64, size: u64) -> Result<Bytes, Error> {
        self.inner.load_bytes(self.translate(addr), size)
    }

    fn lr(&self) -> &Self::REG {
        // LR stores guest addresses (not translated) because the comparison
        // in SC instruction uses guest addresses
        self.inner.lr()
    }

    fn set_lr(&mut self, value: &Self::REG) {
        // Don't translate - LR stores guest addresses for comparison
        self.inner.set_lr(value)
    }
}
